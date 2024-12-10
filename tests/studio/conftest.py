import base64
from textwrap import dedent
from unittest.mock import MagicMock

import pytest

from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.importers.importer import FlowSyncImporter, TrainingDataImporter
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string, read_yaml_file
from rasa.studio.auth import StudioAuth
from rasa.studio.config import StudioConfig
from rasa.studio.upload import DOMAIN_KEYS, extract_values


@pytest.fixture
def mock_keycloak_instance_object() -> MagicMock:
    keycloak_open_id_instance = MagicMock()
    keycloak_open_id_instance.server_url = "http://localhost:8080"
    keycloak_open_id_instance.client_id = "client_id"
    keycloak_open_id_instance.realm_name = "realm_name"
    keycloak_open_id_instance.token = MagicMock()
    return keycloak_open_id_instance


@pytest.fixture
def mock_keycloak_instance(
    mock_keycloak_open_id: MagicMock, mock_keycloak_instance_object: MagicMock
) -> MagicMock:
    mock_keycloak_open_id.return_value = mock_keycloak_instance_object
    return mock_keycloak_instance_object


@pytest.fixture
def studio_auth(mock_keycloak_instance: MagicMock) -> StudioAuth:
    config = StudioConfig(
        authentication_server_url="http://localhost:8080",
        studio_url="http://localhost:8080/graphql",
        client_id="client_id",
        realm_name="realm_name",
    )
    auth = StudioAuth(config)
    return auth


CALM_CUSTOMIZED_PATTERNS_YAML = dedent(
    """\
    flows:
      pattern_completed:
        steps:
        - id: action_listen
          next: END
          metadata:
            line_numbers: 4-7
          action: action_listen
        name: pattern_completed
        description: This is a pattern
        file_path: data/upload/customized_default_flows.yml
    """
)

CALM_NLU_YAML = dedent(
    """\
    version: "3.1"
    nlu:
    - intent: health_advice
      examples: |
        - I need some medical advice.
        - Can you help me with some health issues?
        - I need medical support.
        - I'm experiencing some symptoms and I need guidance on what to do.
        - Can you provide me with health recommendations?
        - I'm struggling with some health concerns. Can you offer advice?
        - Can you suggest ways to improve my overall well-being?
        - I'm looking for tips on managing stress and anxiety. Any advice?
        - I have a specific health question. Can you offer me some insights?
        - I need suggestions on maintaining a healthy diet and exercise routine.
        - Is there anyone knowledgeable about natural remedies who can give me advice?
        - Can you provide me with information on preventing common illnesses?
        - I'm interested in learning about alternative therapies. Can you share your expertise?
        - Can you recommend a good doctor? I'm not feeling well.
    """  # noqa: E501
)

CALM_DOMAIN_DIRECTORY_YAML = dedent(
    "version: '3.1'\nresponses:\n  utter_goodbye:\n  - text: Goodbye!\n  utter_greet:\n  - text: Hello {name}!\nslots:\n  name:\n    type: text\nsession_config:\n  session_expiration_time: 60\n  carry_over_slots_to_new_session: true\n"  # noqa: E501
)
DOMAIN_DIRECTORY_FLOWS_YAML = dedent(
    "flows:\n  greeting:\n    steps:\n    - id: 0_utter_greet\n      next: END\n      metadata:\n        line_numbers: 6-6\n      action: utter_greet\n    name: greeting\n    description: Greet the user\n    file_path: data/upload/simple_bot_with_domain_directory/data/flows.yml\n"  # noqa: E501
)

CALM_ENDPOINTS_YAML = "nlg: \ntype: rephrase\n"


def encode_yaml(yaml: str):
    return base64.b64encode(yaml.encode("utf-8")).decode("utf-8")


def get_calm_domain_yaml(domain_path: str) -> str:
    importer = TrainingDataImporter.load_from_dict(
        domain_path=domain_path,
    )
    domain = importer.get_user_domain().as_dict()
    domain = extract_values(domain, DOMAIN_KEYS)
    return dump_obj_as_yaml_to_string(domain)


def get_calm_config_yaml(config_path: str) -> str:
    return dump_obj_as_yaml_to_string(read_yaml_file(config_path))


def get_flows_yaml(flows_path: str) -> str:
    flow_importer = FlowSyncImporter.load_from_dict(
        training_data_paths=[flows_path],
    )
    flows = list(flow_importer.get_user_flows())
    return YamlFlowsWriter().dumps(flows)


def mock_questionary_text(question, default=""):
    return MagicMock(ask=lambda: "test")
