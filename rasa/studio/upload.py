import argparse
import base64
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Text, Tuple, Union

import questionary
import requests
import structlog

import rasa.cli.telemetry
import rasa.cli.utils
import rasa.shared.utils.cli
import rasa.shared.utils.io
from rasa.cli.validation.config_path_validation import get_validated_path
from rasa.shared.constants import (
    CONFIG_LANGUAGE_KEY,
    CONFIG_LLM_KEY,
    CONFIG_MODEL_NAME_KEY,
    CONFIG_PIPELINE_KEY,
    CONFIG_POLICIES_KEY,
    CONFIG_RECIPE_KEY,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATHS,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.importers.utils import (
    CALMUserData,
    extract_calm_import_parts_from_importer,
)
from rasa.shared.nlu.training_data.formats.rasa_yaml import (
    RasaYAMLReader,
    RasaYAMLWriter,
)
from rasa.shared.utils.llm import collect_custom_prompts
from rasa.shared.utils.yaml import (
    dump_obj_as_yaml_to_string,
    read_yaml_file,
)
from rasa.studio import results_logger
from rasa.studio.auth import KeycloakTokenReader
from rasa.studio.config import StudioConfig
from rasa.studio.results_logger import StudioResult, with_studio_error_handler
from rasa.studio.utils import validate_argument_paths
from rasa.telemetry import track_upload_to_studio_failed
from rasa.utils.json_utils import extract_values

structlogger = structlog.get_logger()

CONFIG_KEYS = [
    "recipe",
    "language",
    "pipeline",
    "llm",
    "policies",
    "model_name",
    "assistant_id",
]

DOMAIN_KEYS = [
    "version",
    "actions",
    "responses",
    "slots",
    "intents",
    "entities",
    "forms",
    "session_config",
]


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


def run_validation(args: argparse.Namespace) -> None:
    """Run the validation before uploading to Studio.

    This is to avoid uploading invalid assistant data
    that would raise errors during Rasa Pro training in Studio.

    The validation checks that were selected to be run before uploading
    maintain parity with the features that are supported in Studio.
    """
    from rasa.validator import Validator

    training_data_paths = args.data if isinstance(args.data, list) else [args.data]
    training_data_importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain,
        training_data_paths=training_data_paths,
        config_path=args.config,
        expand_env_vars=False,
    )

    structlogger.info(
        "rasa.studio.upload.validating_data",
        event_info="Validating domain and training data...",
    )

    validator = Validator.from_importer(training_data_importer)

    if not validator.verify_studio_supported_validations():
        structlogger.error(
            "rasa.studio.upload.validate_files.project_validation_error",
            event_info="Project validation completed with errors.",
        )
        sys.exit(1)

    structlogger.info(
        "rasa.studio.upload.validate_files.success",
        event_info="Project validation completed successfully.",
    )


def handle_upload(args: argparse.Namespace) -> None:
    """Uploads primitives to rasa studio."""
    validate_argument_paths(args)
    studio_config = StudioConfig.read_config()
    endpoint = studio_config.studio_url
    verify = not studio_config.disable_verify

    if not endpoint:
        rasa.shared.utils.cli.print_error_and_exit(
            "No GraphQL endpoint found in config. Please run `rasa studio config`."
        )

    if not is_auth_working(endpoint, verify):
        rasa.shared.utils.cli.print_error_and_exit(
            "Authentication is invalid or expired. Please run `rasa studio login`."
        )

    from rasa.core.config.configuration import Configuration

    Configuration.initialise_empty()

    structlogger.info("rasa.studio.upload.loading_data", event_info="Loading data...")

    args.domain = get_validated_path(args.domain, "domain", DEFAULT_DOMAIN_PATHS)

    args.config = get_validated_path(args.config, "config", DEFAULT_CONFIG_PATH)

    config = read_yaml_file(args.config, expand_env_vars=False)
    assistant_name = args.assistant_name or _get_assistant_name(config)
    args.assistant_name = assistant_name
    if not _handle_existing_assistant(
        assistant_name, studio_config.studio_url, verify, args
    ):
        return

    Domain.expand_env_vars = False
    RasaYAMLReader.expand_env_vars = False
    YAMLFlowsReader.expand_env_vars = False

    upload_calm_assistant(args, endpoint, verify=verify)


config_keys = [
    CONFIG_RECIPE_KEY,
    CONFIG_POLICIES_KEY,
    CONFIG_PIPELINE_KEY,
    CONFIG_LANGUAGE_KEY,
    CONFIG_LLM_KEY,
    CONFIG_MODEL_NAME_KEY,
]


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

    structlogger.info(
        "rasa.studio.upload.name_selected",
        event_info=f"Uploading assistant with the name '{assistant_name}'.",
        assistant_name=assistant_name,
    )
    return assistant_name


def build_calm_import_parts(
    data_path: Union[Text, List[Text]],
    domain_path: Text,
    config_path: Text,
    endpoints_path: Optional[Text] = None,
    assistant_name: Optional[Text] = None,
) -> Tuple[str, CALMUserData]:
    """Builds the parts of the assistant to be uploaded to Studio.

    Args:
        data_path: The path to the training data
        domain_path: The path to the domain
        config_path: The path to the config
        endpoints_path: The path to the endpoints
        assistant_name: The name of the assistant

    Returns:
        The assistant name and the parts to be uploaded
    """
    training_data_paths = data_path if isinstance(data_path, list) else [str(data_path)]
    importer = TrainingDataImporter.load_from_dict(
        domain_path=domain_path,
        config_path=config_path,
        training_data_paths=training_data_paths,
        expand_env_vars=False,
    )

    config = read_yaml_file(config_path, expand_env_vars=False)
    endpoints = read_yaml_file(endpoints_path, expand_env_vars=False)
    assistant_name = assistant_name or _get_assistant_name(config)

    parts = extract_calm_import_parts_from_importer(
        importer=importer,
        config=config,
        endpoints=endpoints,
    )

    return assistant_name, parts


@with_studio_error_handler
def upload_calm_assistant(
    args: argparse.Namespace, endpoint: str, verify: bool = True
) -> StudioResult:
    def yaml_or_empty(part: Dict[Text, Any]) -> Optional[str]:
        return dump_obj_as_yaml_to_string(part) if part else None

    run_validation(args)
    structlogger.info(
        "rasa.studio.upload.loading_data", event_info="Parsing CALM assistant data..."
    )
    assistant_name, parts = build_calm_import_parts(
        data_path=args.data,
        domain_path=args.domain,
        config_path=args.config,
        endpoints_path=args.endpoints,
        assistant_name=args.assistant_name,
    )

    prompts_json = collect_custom_prompts(parts.config, parts.endpoints)
    graphql_req = build_import_request(
        assistant_name,
        flows_yaml=yaml_or_empty(parts.flows),
        domain_yaml=yaml_or_empty(parts.domain),
        config_yaml=yaml_or_empty(parts.config),
        endpoints=yaml_or_empty(parts.endpoints),
        nlu_yaml=yaml_or_empty(parts.nlu),
        prompts_json=prompts_json,
    )

    structlogger.info(
        "rasa.studio.upload.calm", event_info="Uploading to Rasa Studio..."
    )
    return make_request(endpoint, graphql_req, verify)


@with_studio_error_handler
def upload_nlu_assistant(
    args: argparse.Namespace, endpoint: str, verify: bool = True
) -> StudioResult:
    """Uploads the classic (dm1) assistant data to Rasa Studio.

    Args:
        args: The command line arguments
            - data: The path to the training data
            - domain: The path to the domain
            - intents: The intents to upload
            - entities: The entities to upload
        endpoint: The studio endpoint
        verify: Whether to verify SSL
    Returns:
        None
    """
    structlogger.info(
        "rasa.studio.upload.nlu_data_read",
        event_info="Found DM1 assistant data, parsing...",
    )
    training_data_paths = args.data if isinstance(args.data, list) else [args.data]
    importer = TrainingDataImporter.load_from_dict(
        domain_path=args.domain,
        training_data_paths=training_data_paths,
        config_path=args.config,
        expand_env_vars=False,
    )

    intents_from_files = importer.get_nlu_data().intents

    domain_from_files = importer.get_domain()
    entities_from_files = domain_from_files.entities
    entities, intents = _get_selected_entities_and_intents(
        args, intents_from_files, entities_from_files
    )

    config_from_files = importer.get_config()
    config = extract_values(config_from_files, CONFIG_KEYS)

    assistant_name = _get_assistant_name(config)

    structlogger.info(
        "rasa.studio.upload.nlu_data_validate", event_info="Validating data..."
    )
    _check_for_missing_primitives(
        intents, entities, intents_from_files, entities_from_files
    )

    nlu_examples = importer.get_nlu_data().filter_training_examples(
        lambda ex: ex.get("intent") in intents
    )

    all_entities = _add_missing_entities(nlu_examples.entities, entities)
    nlu_examples_yaml = RasaYAMLWriter().dumps(nlu_examples)

    domain = _filter_domain(all_entities, intents, domain_from_files.as_dict())
    domain_yaml = dump_obj_as_yaml_to_string(domain)

    graphql_req = build_request(assistant_name, nlu_examples_yaml, domain_yaml)

    structlogger.info(
        "rasa.studio.upload.nlu", event_info="Uploading to Rasa Studio..."
    )
    return make_request(endpoint, graphql_req, verify)


def is_auth_working(endpoint: str, verify: bool = True) -> bool:
    """Send a test request to Studio to check if auth is working."""
    result = make_request(
        endpoint,
        {
            "operationName": "LicenseDetails",
            "query": (
                "query LicenseDetails {\n"
                "  licenseDetails {\n"
                "    valid\n"
                "    scopes\n"
                "    __typename\n"
                "  }\n"
                "}"
            ),
            "variables": {},
        },
        verify,
    )
    return result.was_successful


def make_request(endpoint: str, graphql_req: Dict, verify: bool = True) -> StudioResult:
    """Makes a request to the studio endpoint to upload data.

    Args:
        endpoint: The studio endpoint
        graphql_req: The graphql request
        verify: Whether to verify SSL
    """
    token = KeycloakTokenReader().get_token()
    res = requests.post(
        endpoint,
        json=graphql_req,
        headers={
            "Authorization": f"{token.token_type} {token.access_token}",
            "Content-Type": "application/json",
        },
        verify=verify,
    )
    if results_logger.response_has_errors(res.json()):
        track_upload_to_studio_failed(res.json())
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
                "rasa.studio.upload.adding_missing_entity",
                event_info=(
                    f"Adding entity '{entity}' to upload "
                    "since it is used in an intent."
                ),
                entity=entity,
            )
            all_entities.append(entity)
    return all_entities


def build_import_request(
    assistant_name: str,
    flows_yaml: Optional[str] = None,
    domain_yaml: Optional[str] = None,
    config_yaml: Optional[str] = None,
    endpoints: Optional[str] = None,
    nlu_yaml: Optional[str] = None,
    prompts_json: Optional[Dict[str, str]] = None,
) -> Dict:
    """Builds the GraphQL request for uploading a modern assistant.

    Args:
        assistant_name: The name of the assistant
        flows_yaml: The YAML representation of the flows
        domain_yaml: The YAML representation of the domain
        config_yaml: The YAML representation of the config
        endpoints: The YAML representation of the endpoints
        nlu_yaml: The YAML representation of the NLU data
        prompts_json: The JSON representation of the prompts

    Returns:
        A dictionary representing the GraphQL request for uploading the assistant.
    """
    inputs_map = {
        "domain": domain_yaml,
        "flows": flows_yaml,
        "config": config_yaml,
        "endpoints": endpoints,
        "nlu": nlu_yaml,
    }

    payload: Dict[Text, Any] = {
        field: convert_string_to_base64(value)
        for field, value in inputs_map.items()
        if value is not None
    }

    if prompts_json:
        payload["prompts"] = prompts_json

    variables_input = {"assistantName": assistant_name, **payload}

    graphql_req = {
        "query": (
            "mutation UploadModernAssistant($input: UploadModernAssistantInput!)"
            "{\n  uploadModernAssistant(input: $input)\n}"
        ),
        "variables": {"input": variables_input},
    }

    return graphql_req


def convert_string_to_base64(string: str) -> str:
    """Converts a string to base64.

    Args:
        string: The string to convert

    Returns:
        The base64 encoded string
    """
    return base64.b64encode(string.encode("utf-8")).decode("utf-8")


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
    entities: List[Union[str, Dict]],
    intents: List[str],
    domain_from_files: Dict[str, Any],
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


def remove_quotes(node: Any) -> Any:
    """Transform function to remove quotes from a node if it is a string.

    This is to prevent wrapping unexpanded environment variables in quotes
    when uploading endpoints to Rasa Studio.
    """
    if isinstance(node, str):
        matches = re.findall(r"'\$\{([^}]+)\}'", node)
        for match in matches:
            node = node.replace(f"'${{{match}}}'", f"${{{match}}}")
        return node
    elif isinstance(node, dict):
        return {k: remove_quotes(v) for k, v in node.items()}
    else:
        return node


def check_if_assistant_already_exists(
    assistant_name: str, endpoint: str, verify: bool = True
) -> bool:
    """Checks if the assistant already exists in Studio.

    Args:
        assistant_name: The name of the assistant
        endpoint: The studio endpoint
        verify: Whether to verify SSL

    Returns:
        bool: The upload confirmation
    """
    graphql_req = build_get_assistant_by_name_request(assistant_name)

    structlogger.info(
        "rasa.studio.upload.assistant_already_exists",
        event_info="Checking if assistant already exists...",
        assistant_name=assistant_name,
    )

    token = KeycloakTokenReader().get_token()
    res = requests.post(
        endpoint,
        json=graphql_req,
        headers={
            "Authorization": f"{token.token_type} {token.access_token}",
            "Content-Type": "application/json",
        },
        verify=verify,
    )
    response = res.json()["data"]["assistantByName"] or {}
    if results_logger.response_has_id(response):
        structlogger.info(
            "rasa.studio.upload.assistant_already_exists",
            event_info="Assistant already exists.",
        )
        return True

    structlogger.info(
        "rasa.studio.upload.assistant_already_exists", event_info="Assistant not found."
    )
    return False


def build_get_assistant_by_name_request(
    assistant_name: str,
) -> Dict:
    graphql_req = {
        "query": (
            "query AssistantByName($input: AssistantByNameInput!) {"
            " assistantByName(input: $input) {"
            " ... on Assistant { id name mode }"
            " ... on AssistantByName_AssistantNotFound { _ }"
            " }"
            "}"
        ),
        "variables": {
            "input": {
                "assistantName": assistant_name,
            }
        },
    }
    return graphql_req


def _handle_existing_assistant(
    assistant_name: str,
    endpoint: str,
    verify: bool,
    args: argparse.Namespace,
) -> bool:
    """Deal with the case that an assistant with the same name already exists.

    Args:
        assistant_name: The name of the assistant
        endpoint: The studio endpoint
        verify: Whether to verify SSL
        args: The command line arguments

    Returns:
        bool: True if the assistant does not exist and can be created,
              False if the assistant already exists and was linked.
    """
    from rasa.studio.link import handle_link

    if not check_if_assistant_already_exists(assistant_name, endpoint, verify):
        return True

    should_link = questionary.confirm(
        f"An assistant named {assistant_name} already exists in Studio. "
        f"Would you like to link your local project to this existing assistant?"
    ).ask()

    if not should_link:
        rasa.shared.utils.cli.print_error_and_exit("Upload cancelled.")
        return False

    args.assistant_name = assistant_name
    handle_link(args)
    return False
