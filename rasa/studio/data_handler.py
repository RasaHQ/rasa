import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rasa.shared.utils.cli
import requests
from rasa.shared.core.domain import KEY_RESPONSES, KEY_SLOTS, Domain
from rasa.shared.core.flows.flow import Flow
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml, write_yaml
from rasa.utils.common import get_temp_dir_name

from rasa.studio.auth import KeycloakToken, KeycloakTokenReader, StudioAuth
from rasa.studio.config import StudioConfig

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
        request = {
            "query": (
                "query ExportAsEncodedYaml($input: ExportAsEncodedYamlInput!) "
                "{ exportAsEncodedYaml(input: $input) "
                "{ ... on ExportModernAsEncodedYamlOutput "
                "{ nlu flows domain } "
                "... on ExportClassicAsEncodedYamlOutput "
                "{ nlu domain }}}"
            ),
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

    def _make_request(self, GQL_req: Dict[Any, Any]) -> Dict[Any, Any]:
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
        response = self._make_request(GQL_req)
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
        response = self._make_request(GQL_req)
        self._extract_data(response)

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

        if not self.has_nlu() and not self.has_flows():
            raise RasaException("No nlu or flows data in Studio response.")

    @staticmethod
    def _decode_response(data: str) -> Optional[str]:
        if not data:
            return None
        return base64.b64decode(data).decode("utf-8")


class DataDiffGenerator:
    """Generate diff between original and studio data."""

    def __init__(
        self,
        original_domain: Optional[Dict] = None,
        studio_domain: Optional[Dict] = None,
        original_nlu: Optional[Dict[str, List]] = None,
        studio_nlu: Optional[Dict[str, List]] = None,
        original_flows: Optional[List[Flow]] = None,
        studio_flows: Optional[List[Flow]] = None,
    ):
        self.original_domain = original_domain
        self.studio_domain = studio_domain
        self.original_nlu = original_nlu
        self.studio_nlu = studio_nlu
        self.original_flows = original_flows
        self.studio_flows = studio_flows

    def create_new_domain_from_diff(self) -> Dict:
        """Create a new domain file from the diff."""
        if self.studio_domain is None or self.original_domain is None:
            return {}
        return self._domain_from_diff_rec(self.studio_domain, self.original_domain)

    @staticmethod
    def _domain_from_diff_rec(studio: Dict, original: Dict) -> Dict:
        ret_dict = {}
        for key in studio:
            if key not in original:
                ret_dict[key] = studio[key]
            elif isinstance(studio[key], dict):
                ret_dict[key] = DataDiffGenerator._domain_from_diff_rec(
                    studio[key], original[key]
                )
                # remove empty diffs
                if not ret_dict[key]:
                    del ret_dict[key]
                elif key not in [KEY_SLOTS, KEY_RESPONSES]:
                    # copy over the whole key not just the diff in case of items
                    # to get complete (valid) data not just the diff
                    ret_dict[key] = studio[key]
            elif isinstance(studio[key], list):
                ret_dict[key] = []
                for item in studio[key]:
                    if item not in original[key]:
                        ret_dict[key].append(item)

                # if list is empty, remove it
                if not ret_dict[key]:
                    del ret_dict[key]

        return ret_dict

    def create_new_nlu_from_diff(self) -> Dict:
        """Create a new nlu file from the diff."""
        if self.studio_nlu is None or self.original_nlu is None:
            return {"nlu": []}

        nlu_diff = []
        nlu_new_examples = [
            new for new in self.studio_nlu["nlu"] if new not in self.original_nlu["nlu"]
        ]
        if nlu_new_examples:
            for new_example in nlu_new_examples:
                nlu_diff.append(new_example)
                intent = new_example.get("intent")
                if intent:
                    self._diff_nlu_examples(new_example, nlu_diff, "intent", intent)
                synonym = new_example.get("synonym")
                if synonym:
                    self._diff_nlu_examples(new_example, nlu_diff, "synonym", synonym)

        return {"nlu": nlu_diff}

    def create_new_flows_from_diff(self) -> List[Flow]:
        """Create a new flows file from the diff."""
        if self.studio_flows is None or self.original_flows is None:
            return []

        flows_new = [new for new in self.studio_flows if new not in self.original_flows]
        return flows_new

    def _diff_nlu_examples(
        self, new_example: Dict, nlu_diff: List, match_key: str, match_value: str
    ) -> None:
        """Creates diff of nlu data examples.

        Args:
            new_example (Dict): intent or synonym with examples
            nlu_diff (List): list of diff examples
            match_key (str): intent or synonym name
            match_value (str): intent or synonym value
        """
        orig = list(
            filter(
                lambda x: x.get(match_key) == match_value,
                self.original_nlu["nlu"],  # type: ignore[index]
            )
        )
        if len(orig) == 1:
            orig_ex = orig[0]["examples"].split("\n")
            new_ex = new_example["examples"].split("\n")
            new_example["examples"] = "\n".join(
                list(set(new_ex).difference(set(orig_ex)))
            )
            if not new_example["examples"]:
                nlu_diff.remove(new_example)


def import_data_from_studio(
    handler: StudioDataHandler, domain_path: Path, data_paths: List[Path]
) -> Tuple[TrainingDataImporter, TrainingDataImporter]:
    """Construct TrainingDataImporter from Studio data and original data.

    Args:
        handler (StudioDataHandler): handler with data from studio
        domain_path (Path): Path to a domain file
        data_paths (List[Path]): List of paths to training data files

    Returns:
        Tuple[TrainingDataImporter, TrainingDataImporter]:
        data from studio and original data
    """
    tmp_dir = get_temp_dir_name()
    data_original = TrainingDataImporter.load_from_dict(
        domain_path=domain_path, training_data_paths=data_paths
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
        domain_path=str(domain_file), training_data_paths=data_paths
    )

    return data_from_studio, data_original
