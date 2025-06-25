import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import rasa.shared.utils.cli
from rasa.shared.core.domain import KEY_RESPONSES, KEY_SLOTS, Domain
from rasa.shared.core.flows.flow import Flow
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml, write_yaml
from rasa.studio.auth import KeycloakToken, KeycloakTokenReader, StudioAuth
from rasa.studio.config import StudioConfig
from rasa.utils.common import get_temp_dir_name

logger = logging.getLogger(__name__)


class StudioDataHandler:
    def __init__(
        self, studio_config: StudioConfig, assistant_name: Optional[str]
    ) -> None:
        self.studio_config = studio_config.read_config()
        if not assistant_name:
            rasa.shared.utils.cli.print_error_and_exit(
                "Assistant name not specified please specify"
                "the assistant name with: 'rasa studio <command> assistant_name'"
            )
        self.assistant_name = assistant_name

        self.nlu: Optional[str] = None
        self.domain: Optional[str] = None
        self.flows: Optional[str] = None

    def has_flows(self) -> bool:
        return self.flows is not None

    def has_nlu(self) -> bool:
        return self.nlu is not None

    def _build_request(
        self,
        intent_names: Optional[List[str]] = None,
        entity_names: Optional[List[str]] = None,
    ) -> dict:
        from rasa.studio.prompts import (
            COMMAND_GENERATOR_NAME,
            CONTEXTUAL_RESPONSE_REPHRASER_NAME,
            ENTERPRISE_SEARCH_NAME,
        )

        request = {
            "query": "query ExportAsEncodedYaml($input: ExportAsEncodedYamlInput!) {\n"
            "  exportAsEncodedYaml(input: $input) {\n"
            "    ... on ExportModernAsEncodedYamlOutput {\n"
            "      nlu\n"
            "      flows\n"
            "      domain\n"
            "      endpoints\n"
            "      config\n"
            "      prompts {\n"
            f"        {COMMAND_GENERATOR_NAME}\n"
            f"        {CONTEXTUAL_RESPONSE_REPHRASER_NAME}\n"
            f"        {ENTERPRISE_SEARCH_NAME}\n"
            "      }\n"
            "    }\n"
            "    ... on ExportClassicAsEncodedYamlOutput {\n"
            "      nlu\n"
            "      domain\n"
            "    }\n"
            "  }\n"
            "}\n",
            "variables": {"input": {"assistantName": self.assistant_name}},
        }
        if intent_names or entity_names:
            obj = []
            if intent_names:
                obj.append(
                    {
                        "name": intent_names,
                        "type": "Intent",
                    }
                )
            if entity_names:
                obj.append(
                    {
                        "name": entity_names,
                        "type": "Entity",
                    }
                )
            request["variables"]["input"]["objects"] = obj  # type: ignore[index]

        return request

    def _make_request(
        self, GQL_req: Dict[Any, Any], verify: bool = True
    ) -> Dict[Any, Any]:
        token = KeycloakTokenReader().get_token()
        if token.is_expired():
            token = self.refresh_token(token)

        if not self.studio_config.studio_url:
            return rasa.shared.utils.cli.print_error_and_exit(
                "No endpoint found in config. Please run `rasa studio config`."
            )

        res = requests.post(
            url=self.studio_config.studio_url,
            json=GQL_req,
            headers={
                "Authorization": f"{token.token_type} {token.access_token}",
                "Content-Type": "application/json",
            },
            verify=verify,
        )
        if res.status_code != 200:
            raise RasaException(
                f"Download from Studio with URL: "
                f"{self.studio_config.studio_url} failed "
                f"with status code {res.status_code}"
            )

        res_json = res.json()
        if self._validate_response(res_json):
            return res_json

        return rasa.shared.utils.cli.print_error_and_exit(
            "Failed to download data from Rasa Studio."
        )

    def refresh_token(self, token: KeycloakToken) -> KeycloakToken:
        if not token.can_refresh():
            rasa.shared.utils.cli.print_error_and_exit(
                "Access token expired and cannot be refreshed. "
                "Please run `rasa studio login`."
            )
        auth = StudioAuth(self.studio_config)
        auth.refresh_token(token.refresh_token)
        token = KeycloakTokenReader().get_token()
        return token

    def request_all_data(self) -> None:
        """Gets the data from Rasa Studio.

        Returns:
            The data from Rasa Studio.
        """
        GQL_req = self._build_request()
        verify = not self.studio_config.disable_verify

        response = self._make_request(GQL_req, verify=verify)
        self._extract_data(response)

    def request_data(
        self, intent_names: List[str], entity_names: List[str], **kwargs: Any
    ) -> None:
        """Gets the data from Rasa Studio.

        Args:
            intent_names: List of intents to download
            entity_names: List of entities to download
            **kwargs: Additional arguments to pass to the request

        Returns:
            The data from Rasa Studio.
        """
        GQL_req = self._build_request(intent_names, entity_names)
        verify = not self.studio_config.disable_verify

        response = self._make_request(GQL_req, verify=verify)
        self._extract_data(response)

    def get_config(self) -> Optional[str]:
        return self.config

    def get_endpoints(self) -> Optional[str]:
        return self.endpoints

    def get_prompts(self) -> Optional[dict]:
        return self.prompts

    def _validate_response(self, response: dict) -> bool:
        """Validates the response from Rasa Studio.

        Args:
            response: The response from Rasa Studio.

        Returns:
            True if the response is valid, False otherwise.
        """
        try:
            if not response.get("data"):
                for error in response["errors"]:
                    logger.error(error["message"])
                return False
            if not response["data"]["exportAsEncodedYaml"]:
                logger.error(response["data"])
                return False
            if not (
                response["data"]["exportAsEncodedYaml"].get("nlu")
                or response["data"]["exportAsEncodedYaml"].get("flows")
            ):
                logger.error(
                    "No nlu or flows data in Studio response."
                    "Cannot determine assistant type."
                )
                return False
        except KeyError:
            return False
        return True

    def _extract_data(self, response: dict) -> None:
        return_data = response["data"]["exportAsEncodedYaml"]

        self.nlu = self._decode_response(return_data.get("nlu"))
        self.domain = self._decode_response(return_data.get("domain"))
        self.flows = self._decode_response(return_data.get("flows"))
        self.config = self._decode_response(return_data.get("config"))
        self.endpoints = self._decode_response(return_data.get("endpoints"))
        self.prompts = return_data.get("prompts")

        if not self.has_nlu() and not self.has_flows():
            raise RasaException("No nlu or flows data in Studio response.")

    @staticmethod
    def _decode_response(data: str) -> Optional[str]:
        if not data:
            return None
        return base64.b64decode(data).decode("utf-8")


def combine_domains(
    studio_domain: Dict[str, Any], original_domain: Dict[str, Any]
) -> Dict:
    """Create a new domain file from the diff."""
    if studio_domain is None or original_domain is None:
        return {}
    return _combine_domain_keys(studio_domain, original_domain)


def _combine_domain_keys(
    first_domain: Dict[str, Any], second_domain: Dict[str, Any]
) -> Dict[str, Any]:
    combined_keys = {}
    for key in first_domain:
        if key not in second_domain:
            combined_keys[key] = first_domain[key]
        elif isinstance(first_domain[key], dict):
            combined_keys[key] = _combine_domain_keys(
                first_domain[key], second_domain[key]
            )
            # remove empty diffs
            if not combined_keys[key]:
                del combined_keys[key]
            elif key not in [KEY_SLOTS, KEY_RESPONSES]:
                # for all keys except slots and responses, we want to keep the
                # keys from the first domain
                combined_keys[key] = first_domain[key]
        elif isinstance(first_domain[key], list):
            combined_keys[key] = []
            for item in first_domain[key]:
                if item not in second_domain[key]:
                    combined_keys[key].append(item)

            # if list is empty, remove it
            if not combined_keys[key]:
                del combined_keys[key]

    return combined_keys


def _diff_nlu_examples(
    new_example: Dict,
    nlu_diff: List,
    match_key: str,
    match_value: str,
    original_nlu_examples: List,
) -> None:
    """Creates diff of nlu data examples.

    Args:
        new_example (Dict): intent or synonym with examples
        nlu_diff (List): list of diff examples
        match_key (str): intent or synonym name
        match_value (str): intent or synonym value
        original_nlu_examples (List): original nlu examples
    """
    orig = list(
        filter(
            lambda x: x.get(match_key) == match_value,
            original_nlu_examples,
        )
    )
    if len(orig) == 1:
        orig_ex = orig[0]["examples"].split("\n")
        new_ex = new_example["examples"].split("\n")
        new_example["examples"] = "\n".join(list(set(new_ex).difference(set(orig_ex))))
        if not new_example["examples"]:
            nlu_diff.remove(new_example)


def create_new_nlu_from_diff(
    studio_nlu: Dict[str, Any], original_nlu: Dict[str, Any]
) -> Dict:
    """Create a new nlu file from the diff."""
    # `or []` handles the case where the data contains the property as an empty
    # key, example yaml:
    # ```
    # nlu:
    # ```
    # in this case, the yaml parser will return an empty dict (because it
    # can't know that it is supposed to be a list, so we need to convert it
    # to a list
    studio_nlu_data = studio_nlu.get("nlu", []) or []
    original_nlu_data = original_nlu.get("nlu", []) or []

    nlu_diff = []
    for new in studio_nlu_data:
        if new in original_nlu_data:
            continue

        nlu_diff.append(new)
        intent = new.get("intent")
        if intent:
            _diff_nlu_examples(new, nlu_diff, "intent", intent, original_nlu_data)
        synonym = new.get("synonym")
        if synonym:
            _diff_nlu_examples(new, nlu_diff, "synonym", synonym, original_nlu_data)

    return {"nlu": nlu_diff}


def create_new_flows_from_diff(
    studio_flows: List[Flow], original_flows: List[Flow]
) -> List[Flow]:
    """Create a new flows file from the diff."""
    flows_new = [new for new in studio_flows if new not in original_flows]
    return flows_new


def import_data_from_studio(
    handler: StudioDataHandler, domain_path: Path, data_path: Path
) -> Tuple[TrainingDataImporter, TrainingDataImporter]:
    """Construct TrainingDataImporter from Studio data and original data.

    Args:
        handler (StudioDataHandler): handler with data from studio
        domain_path (Path): Path to a domain file
        data_path (List[Path]): List of paths to training data files

    Returns:
        Tuple[TrainingDataImporter, TrainingDataImporter]:
        data from studio and original data
    """
    tmp_dir = get_temp_dir_name()
    data_original = TrainingDataImporter.load_from_dict(
        domain_path=str(domain_path), training_data_paths=[str(data_path)]
    )

    data_paths = []

    if handler.has_nlu():
        nlu_file = Path(tmp_dir, "nlu.yml")
        write_yaml(read_yaml(handler.nlu), nlu_file)
        data_paths.append(str(nlu_file))

    if handler.has_flows():
        flows_file = Path(tmp_dir, "flows.yml")
        write_yaml(read_yaml(handler.flows), flows_file)
        data_paths.append(str(flows_file))

    if not data_paths:
        return rasa.shared.utils.cli.print_error_and_exit(
            "No Data for nlu or flows. Can't import data from Studio."
        )

    studio_domain = Domain.from_yaml(handler.domain)
    domain_file = Path(tmp_dir, "domain.yml")
    studio_domain.persist(domain_file)

    data_from_studio = TrainingDataImporter.load_from_dict(
        domain_path=str(domain_file),
        training_data_paths=data_paths,
        expand_env_vars=False,
    )

    return data_from_studio, data_original
