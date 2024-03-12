import argparse
import base64
import logging
from typing import Dict, Iterable, List, Set, Text, Tuple, Union

import rasa.cli.telemetry
import rasa.shared.utils.cli
import rasa.shared.utils.io
import requests
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter
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

    logger.info("Loading data...")
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
    domain = {}
    domain["version"] = domain_from_files["version"]
    domain["intents"] = intents
    domain["entities"] = _remove_not_selected_entities(
        entities, domain_from_files["entities"]
    )

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
