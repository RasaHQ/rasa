import argparse
import base64
import logging
from typing import Dict, Iterable, List, Set, Text, Tuple, Union

import requests

import rasa.cli.telemetry
import rasa.cli.utils
import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.shared.constants import (
    DEFAULT_DOMAIN_PATHS,
)
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter, FlowSyncImporter
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLWriter
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string
from rasa.studio.auth import KeycloakTokenReader
from rasa.studio.config import StudioConfig

logger = logging.getLogger(__name__)


def _get_selected_entities_and_intents(
    args: argparse.Namespace,
    intents_from_files: Set[Text],
    entities_from_files: List[Text],
) -> Tuple[List[Text], List[Text]]:
    entities = args.entities

    if entities is None or len(entities) == 0:
        entities = entities_from_files
        logger.info("No entities specified. Using all entities from files.")

    intents = args.intents

    if intents is None or len(intents) == 0:
        intents = intents_from_files
        logger.info("No intents specified. Using all intents from files.")

    return list(entities), list(intents)


def handle_upload(args: argparse.Namespace) -> None:
    """Uploads primitives to rasa studio."""
    assistant_name = args.assistant_name[0]
    endpoint = StudioConfig.read_config().studio_url
    if not endpoint:
        rasa.shared.utils.cli.print_error_and_exit(
            "No GraphQL endpoint found in config. Please run `rasa studio config`."
        )
    else:
        logger.info("Loading data...")

        args.domain = rasa.cli.utils.get_validated_path(
            args.domain, "domain", DEFAULT_DOMAIN_PATHS
        )

        # check safely if args.calm is set and not fail if not
        if hasattr(args, "calm") and args.calm:
            upload_calm_assistant(args, assistant_name, endpoint)
        else:
            upload_nlu_assistant(args, assistant_name, endpoint)


def extract_values(data: Dict, keys: List[Text]) -> Dict:
    """Extracts values for given keys from a dictionary."""
    return {key: data.get(key) for key in keys if data.get(key)}


def upload_calm_assistant(
    args: argparse.Namespace, assistant_name: str, endpoint: str
) -> None:
    """Uploads the CALM assistant data to Rasa Studio.

    Args:
        args: The command line arguments
            - data: The path to the training data
            - domain: The path to the domain
            - flows: The path to the flows
            - endpoints: The path to the endpoints
            - config: The path to the config
        assistant_name: The name of the assistant
        endpoint: The studio endpoint
    Returns:
        None
    """
    logger.info("Parsing CALM assistant data...")

    try:
        importer = TrainingDataImporter.load_from_dict(
            domain_path=args.domain,
            config_path=args.config,
        )

        # Prepare config and domain
        config_from_files = importer.get_config()
        domain_from_files = importer.get_domain().as_dict()

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
        config_keys = [
            "recipe",
            "language",
            "pipeline",
            "llm",
            "policies",
            "model_name",
        ]

        domain = extract_values(domain_from_files, domain_keys)
        config = extract_values(config_from_files, config_keys)

        training_data_paths = args.data

        if isinstance(training_data_paths, list):
            training_data_paths.append(args.flows)
        elif isinstance(training_data_paths, str):
            training_data_paths = [training_data_paths, args.flows]

        # Prepare flows
        flow_importer = FlowSyncImporter.load_from_dict(
            training_data_paths=training_data_paths
        )

        user_flows = flow_importer.get_flows().user_flows
        flows = list(user_flows)

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
            config_yaml=dump_obj_as_yaml_to_string(config),
            nlu_yaml=nlu_examples_yaml,
        )

        logger.info("Uploading to Rasa Studio...")
        response, status = make_request(endpoint, graphql_req)

        if status:
            rasa.shared.utils.cli.print_success(response)
        else:
            logger.error(f"Failed to upload to Rasa Studio: {response}")
            rasa.shared.utils.cli.print_error(response)

    except Exception as e:
        logger.error(f"An error occurred while uploading the CALM assistant: {e}")


def upload_nlu_assistant(
    args: argparse.Namespace, assistant_name: str, endpoint: str
) -> None:
    """Uploads the classic (dm1) assistant data to Rasa Studio.

    Args:
        args: The command line arguments
            - data: The path to the training data
            - domain: The path to the domain
            - intents: The intents to upload
            - entities: The entities to upload
        assistant_name: The name of the assistant
        endpoint: The studio endpoint
    Returns:
        None
    """
    logger.info("Found DM1 assistant data, parsing...")
    importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain, training_data_paths=args.data
    )

    intents_from_files = importer.get_nlu_data().intents
    entities_from_files = importer.get_domain().entities

    entities, intents = _get_selected_entities_and_intents(
        args, intents_from_files, entities_from_files
    )

    logger.info("Validating data...")
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

    logger.info("Uploading to Rasa Studio...")
    response, status = make_request(endpoint, graphql_req)
    if status:
        rasa.shared.utils.cli.print_success(response)
    else:
        rasa.shared.utils.cli.print_error(response)


def make_request(endpoint: str, graphql_req: Dict) -> Tuple[str, bool]:
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

    response = res.json()

    if _response_has_errors(response):
        print_errors_from_response(response)

    if res.status_code != 200:
        return f"Upload failed with status code {res.status_code}", False
    elif _response_has_errors(response):
        return "Error while uploading data!", False

    return "Upload successful!", True


def _response_has_errors(response: Dict) -> bool:
    return (
        "errors" in response
        and isinstance(response["errors"], list)
        and len(response["errors"]) > 0
    )


def print_errors_from_response(response: Dict) -> None:
    for error in response["errors"]:
        logger.error(error["message"])


def _add_missing_entities(
    entities_from_intents: Iterable[str], entities: List[str]
) -> List[Union[str, Dict]]:
    all_entities: List[Union[str, Dict]] = []
    all_entities.extend(entities)
    for entity in entities_from_intents:
        if entity not in entities:
            logger.warning(
                f"Adding entity '{entity}' to upload since it is used in an intent."
            )
            all_entities.append(entity)
    return all_entities


def build_import_request(
    assistant_name: str,
    flows_yaml: str,
    domain_yaml: str,
    config_yaml: str,
    nlu_yaml: str = "",
) -> Dict:
    # b64encode expects bytes and returns bytes so we need to decode to string
    base64_domain = base64.b64encode(domain_yaml.encode("utf-8")).decode("utf-8")
    base64_flows = base64.b64encode(flows_yaml.encode("utf-8")).decode("utf-8")
    base64_config = base64.b64encode(config_yaml.encode("utf-8")).decode("utf-8")
    base64_nlu = base64.b64encode(nlu_yaml.encode("utf-8")).decode("utf-8")

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
    domain = {
        "version": domain_from_files["version"],
        "intents": intents,
        "entities": _remove_not_selected_entities(
            entities, domain_from_files["entities"]
        ),
    }

    return domain


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
