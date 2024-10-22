import argparse
import base64
import sys
from typing import Dict, Iterable, List, Set, Text, Tuple, Union, Any

import questionary
import requests
import structlog

import rasa.cli.telemetry
import rasa.cli.utils
import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.shared.constants import (
    DEFAULT_DOMAIN_PATHS,
    DEFAULT_CONFIG_PATH,
)
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter, FlowSyncImporter
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string, read_yaml_file
from rasa.studio import results_logger
from rasa.studio.auth import KeycloakTokenReader
from rasa.studio.config import StudioConfig
from rasa.studio.results_logger import StudioResult, with_studio_error_handler

structlogger = structlog.get_logger()


def _get_selected_entities_and_intents(
    args: argparse.Namespace,
    intents_from_files: Set[Text],
    entities_from_files: List[Text],
) -> Tuple[List[Text], List[Text]]:
    entities = args.entities

    if entities is None or len(entities) == 0:
        entities = entities_from_files
        structlogger.info(
            "rasa.studio.upload.entities_empty",
            event_info="No entities specified. Using all entities from files.",
        )

    intents = args.intents

    if intents is None or len(intents) == 0:
        intents = intents_from_files
        structlogger.info(
            "rasa.studio.upload.intents_empty",
            event_info="No intents specified. Using all intents from files.",
        )

    return list(entities), list(intents)


def handle_upload(args: argparse.Namespace) -> None:
    """Uploads primitives to rasa studio."""
    endpoint = StudioConfig.read_config().studio_url
    if not endpoint:
        rasa.shared.utils.cli.print_error_and_exit(
            "No GraphQL endpoint found in config. Please run `rasa studio config`."
        )
    else:
        structlogger.info(
            "rasa.studio.upload.loading_data", event_info="Loading data..."
        )

        args.domain = rasa.cli.utils.get_validated_path(
            args.domain, "domain", DEFAULT_DOMAIN_PATHS
        )

        args.config = rasa.cli.utils.get_validated_path(
            args.config, "config", DEFAULT_CONFIG_PATH
        )

        # check safely if args.calm is set and not fail if not
        if hasattr(args, "calm") and args.calm:
            upload_calm_assistant(args, endpoint)
        else:
            upload_nlu_assistant(args, endpoint)


config_keys = [
    "recipe",
    "language",
    "pipeline",
    "llm",
    "policies",
    "model_name",
    "assistant_id",
]


def extract_values(data: Dict, keys: List[Text]) -> Dict:
    """Extracts values for given keys from a dictionary."""
    return {key: data.get(key) for key in keys if data.get(key)}


def _get_assistant_name(config: Dict[Text, Any]) -> str:
    config_assistant_id = config.get("assistant_id", "")
    assistant_name = questionary.text(
        "Please provide the assistant name", default=config_assistant_id
    ).ask()
    if not assistant_name:
        structlogger.error(
            "rasa.studio.upload.assistant_name_empty",
            event_info="Assistant name cannot be empty.",
        )
        sys.exit(1)

    # if assistant_name exists and different from config assistant_id,
    # notify user and upload with new assistant_name
    same = assistant_name == config_assistant_id
    if not same and config_assistant_id != "":
        structlogger.info(
            "rasa.studio.upload.assistant_name_mismatch",
            event_info=(
                f"Assistant name '{assistant_name}' is different"
                f" from the one in the config file: '{config_assistant_id}'."
            ),
        )

    structlogger.info(f"Uploading assistant with the name '{assistant_name}'.")
    return assistant_name


@with_studio_error_handler
def upload_calm_assistant(args: argparse.Namespace, endpoint: str) -> StudioResult:
    """Uploads the CALM assistant data to Rasa Studio.

    Args:
        args: The command line arguments
            - data: The path to the training data
            - domain: The path to the domain
            - flows: The path to the flows
            - endpoints: The path to the endpoints
            - config: The path to the config
        endpoint: The studio endpoint
    Returns:
        None
    """
    structlogger.info(
        "rasa.studio.upload.loading_data", event_info="Parsing CALM assistant data..."
    )

    importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain,
        config_path=args.config,
    )

    # Prepare config and domain
    config = importer.get_config()
    domain_from_files = importer.get_user_domain().as_dict()
    endpoints_from_files = read_yaml_file(args.endpoints)
    config_from_files = read_yaml_file(args.config)

    # Extract domain and config values
    domain_keys = [
        "version",
        "actions",
        "responses",
        "slots",
        "intents",
        "entities",
        "forms",
        "session_config",
    ]

    domain = extract_values(domain_from_files, domain_keys)

    assistant_name = _get_assistant_name(config)

    training_data_paths = args.data

    if isinstance(training_data_paths, list):
        training_data_paths.append(args.flows)
    elif isinstance(training_data_paths, str):
        if isinstance(args.flows, list):
            training_data_paths = [training_data_paths] + args.flows
        elif isinstance(args.flows, str):
            training_data_paths = [training_data_paths, args.flows]
        else:
            raise RasaException("Invalid flows path")

    # Prepare flows
    flow_importer = FlowSyncImporter.load_from_dict(
        training_data_paths=training_data_paths
    )

    flows = list(flow_importer.get_user_flows())

    # We instantiate the TrainingDataImporter again on purpose to avoid
    # adding patterns to domain's actions. More info https://t.ly/W8uuc
    nlu_importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain, training_data_paths=args.data
    )
    nlu_data = nlu_importer.get_nlu_data()

    intents_from_files = nlu_data.intents

    nlu_examples = nlu_data.filter_training_examples(
        lambda ex: ex.get("intent") in intents_from_files
    )

    nlu_examples_yaml = RasaYAMLWriter().dumps(nlu_examples)

    # Build GraphQL request
    graphql_req = build_import_request(
        assistant_name,
        flows_yaml=YamlFlowsWriter().dumps(flows),
        domain_yaml=dump_obj_as_yaml_to_string(domain),
        config_yaml=dump_obj_as_yaml_to_string(config_from_files),
        endpoints=dump_obj_as_yaml_to_string(endpoints_from_files),
        nlu_yaml=nlu_examples_yaml,
    )

    structlogger.info("Uploading to Rasa Studio...")
    return make_request(endpoint, graphql_req)


@with_studio_error_handler
def upload_nlu_assistant(args: argparse.Namespace, endpoint: str) -> StudioResult:
    """Uploads the classic (dm1) assistant data to Rasa Studio.

    Args:
        args: The command line arguments
            - data: The path to the training data
            - domain: The path to the domain
            - intents: The intents to upload
            - entities: The entities to upload
        endpoint: The studio endpoint
    Returns:
        None
    """
    structlogger.info("Found DM1 assistant data, parsing...")
    importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain, training_data_paths=args.data, config_path=args.config
    )

    intents_from_files = importer.get_nlu_data().intents
    entities_from_files = importer.get_domain().entities

    entities, intents = _get_selected_entities_and_intents(
        args, intents_from_files, entities_from_files
    )

    config_from_files = importer.get_config()
    config = extract_values(config_from_files, config_keys)

    assistant_name = _get_assistant_name(config)

    structlogger.info("Validating data...")
    _check_for_missing_primitives(
        intents, entities, intents_from_files, entities_from_files
    )

    nlu_examples = importer.get_nlu_data().filter_training_examples(
        lambda ex: ex.get("intent") in intents
    )

    all_entities = _add_missing_entities(nlu_examples.entities, entities)
    nlu_examples_yaml = RasaYAMLWriter().dumps(nlu_examples)

    domain = _filter_domain(all_entities, intents, importer.get_domain().as_dict())
    domain_yaml = dump_obj_as_yaml_to_string(domain)

    graphql_req = build_request(assistant_name, nlu_examples_yaml, domain_yaml)

    structlogger.info("Uploading to Rasa Studio...")
    return make_request(endpoint, graphql_req)


def make_request(endpoint: str, graphql_req: Dict) -> StudioResult:
    """Makes a request to the studio endpoint to upload data.

    Args:
        endpoint: The studio endpoint
        graphql_req: The graphql request
    """
    token = KeycloakTokenReader().get_token()
    res = requests.post(
        endpoint,
        json=graphql_req,
        headers={
            "Authorization": f"{token.token_type} {token.access_token}",
            "Content-Type": "application/json",
        },
    )

    if results_logger.response_has_errors(res.json()):
        return StudioResult.error(res.json())
    return StudioResult.success("Upload successful")


def _add_missing_entities(
    entities_from_intents: Iterable[str], entities: List[str]
) -> List[Union[str, Dict]]:
    all_entities: List[Union[str, Dict]] = []
    all_entities.extend(entities)
    for entity in entities_from_intents:
        if entity not in entities:
            structlogger.warning(
                f"Adding entity '{entity}' to upload since it is used in an intent."
            )
            all_entities.append(entity)
    return all_entities


def build_import_request(
    assistant_name: str,
    flows_yaml: str,
    domain_yaml: str,
    config_yaml: str,
    endpoints: str,
    nlu_yaml: str = "",
) -> Dict:
    # b64encode expects bytes and returns bytes so we need to decode to string
    base64_domain = base64.b64encode(domain_yaml.encode("utf-8")).decode("utf-8")
    base64_flows = base64.b64encode(flows_yaml.encode("utf-8")).decode("utf-8")
    base64_config = base64.b64encode(config_yaml.encode("utf-8")).decode("utf-8")
    base64_nlu = base64.b64encode(nlu_yaml.encode("utf-8")).decode("utf-8")
    base64_endpoints = base64.b64encode(endpoints.encode("utf-8")).decode("utf-8")

    graphql_req = {
        "query": (
            "mutation UploadModernAssistant($input: UploadModernAssistantInput!)"
            "{\n  uploadModernAssistant(input: $input)\n}"
        ),
        "variables": {
            "input": {
                "assistantName": assistant_name,
                "domain": base64_domain,
                "flows": base64_flows,
                "nlu": base64_nlu,
                "config": base64_config,
                "endpoints": base64_endpoints,
            }
        },
    }

    return graphql_req


def build_request(
    assistant_name: str, nlu_examples_yaml: str, domain_yaml: str
) -> Dict:
    # b64encode expects bytes and returns bytes so we need to decode to string
    base64_domain = base64.b64encode(domain_yaml.encode("utf-8")).decode("utf-8")
    base64_nlu_examples = base64.b64encode(nlu_examples_yaml.encode("utf-8")).decode(
        "utf-8"
    )

    graphql_req = {
        "query": (
            "mutation ImportFromEncodedYaml($input: ImportFromEncodedYamlInput!)"
            "{\n  importFromEncodedYaml(input: $input)\n}"
        ),
        "variables": {
            "input": {
                "assistantName": assistant_name,
                "domain": base64_domain,
                "nlu": base64_nlu_examples,
            }
        },
    }

    return graphql_req


def _filter_domain(
    entities: List[Union[str, Dict]], intents: List[str], domain_from_files: Dict
) -> Dict:
    """Filters the domain to only include the selected entities and intents."""
    selected_entities = _remove_not_selected_entities(
        entities, domain_from_files.get("entities", [])
    )
    return {
        "version": domain_from_files["version"],
        "intents": intents,
        "entities": selected_entities,
    }


def _check_for_missing_primitives(
    intents: Iterable[str],
    entities: Iterable[str],
    intents_found: Iterable[str],
    entities_found: Iterable[str],
) -> None:
    """Checks if the data contains all intents and entities.

    Args:
        intents: Iterable of intents to check
        entities: Iterable of entities to check
        intents_found: Iterable of intents found in the data
        entities_found: Iterable of entities found in the data
    """
    not_found_entities = [entity for entity in entities if entity not in entities_found]
    not_found_intents = [intent for intent in intents if intent not in intents_found]

    not_found_msg = ""
    if not_found_entities:
        not_found_msg += (
            f"Entities were not found in the training data: {not_found_entities}\n"
        )
    if not_found_intents:
        not_found_msg += (
            f"Intents were not found in the training data: {not_found_intents}\n"
        )

    if not_found_msg:
        raise RasaException(not_found_msg)


def _remove_not_selected_entities(
    entities: List[Union[Text, Dict]], domain_entities: List[Union[Text, Dict]]
) -> List:
    to_remove: List[Union[Text, Dict]] = []

    for entity in domain_entities:
        if isinstance(entity, str) and entity not in entities:
            to_remove.append(entity)
        elif isinstance(entity, dict) and len(entity) == 1:
            entity_name = next(iter(entity))
            if entity_name not in entities:
                to_remove.append(entity)

    for entity in to_remove:
        domain_entities.remove(entity)

    return domain_entities
