import base64
import textwrap
from pathlib import Path
from textwrap import dedent
from typing import Dict, Text
from unittest.mock import MagicMock

import pytest

import rasa.studio.link
import rasa.studio.pull.pull
import rasa.studio.push
import rasa.studio.upload
from rasa.constants import RASA_DIR_NAME
from rasa.shared.core.flows.yaml_flows_io import YamlFlowsWriter
from rasa.shared.importers.importer import FlowSyncImporter, TrainingDataImporter
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string, read_yaml_file
from rasa.studio.auth import StudioAuth
from rasa.studio.config import StudioConfig
from rasa.studio.data_handler import StudioDataHandler
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
        name: pattern_completed
        description: This is a pattern
        file_path: data/upload/customized_default_flows.yml
        steps:
        - id: action_listen
          action: action_listen
          next: END
          metadata:
            line_numbers: 4-7
    """
)

CALM_NLU_YAML = dedent(
    """\
    version: '3.1'
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


def get_calm_domain_yaml(domain_path: Path) -> str:
    importer = TrainingDataImporter.load_from_dict(
        domain_path=str(domain_path),
    )
    domain = importer.get_user_domain().as_dict()
    domain = extract_values(domain, DOMAIN_KEYS)
    return dump_obj_as_yaml_to_string(domain)


def get_calm_config_yaml(config_path: Path) -> str:
    return dump_obj_as_yaml_to_string(read_yaml_file(config_path))


def get_flows_yaml(flows_path: Path) -> str:
    flow_importer = FlowSyncImporter.load_from_dict(
        training_data_paths=[str(flows_path)],
    )
    flows = list(flow_importer.get_user_flows())
    return YamlFlowsWriter().dumps(flows)


def mock_questionary_text(question, default=""):
    return MagicMock(ask=lambda: "test")


@pytest.fixture()
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp Rasa project that is already linked to Studio."""
    monkeypatch.chdir(tmp_path)

    # Link the project as with `rasa studio link`
    (tmp_path / RASA_DIR_NAME).mkdir()
    (tmp_path / RASA_DIR_NAME / "studio.yml").write_text("assistant_name: linked_bot\n")

    # Initialize the project with default files
    (tmp_path / "config.yml").write_text("pipeline: []")
    (tmp_path / "endpoints.yml").write_text("nlg:")
    (tmp_path / "domain.yml").write_text("version: '3.1'")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "nlu.yml").write_text("version: '3.1'\nnlu: []")
    flows_dir = data_dir / "flows"
    flows_dir.mkdir()
    add_contact_flow_yaml = textwrap.dedent(
        """
        flows:
          add_contact:
            name: Add a Contact
            description: Flow to add a contact to your contact list
            steps:
              - action: utter_add_contact
                next: END
        """
    ).lstrip()
    (flows_dir / "add_contact.yml").write_text(add_contact_flow_yaml)
    list_contacts_flow_yaml = textwrap.dedent(
        """
        flows:
          list_contacts:
            name: "list your contacts"
            description: "show your contact list"
            steps:
             - action: list_contacts
             - action: utter_no_contacts
               next: END
        """
    ).lstrip()
    (flows_dir / "list_contacts.yml").write_text(list_contacts_flow_yaml)

    # Mock the read_assistant_name function with project_root as project root
    def read_from_root(*args, **kwargs):
        return rasa.studio.link.read_assistant_name

    monkeypatch.setattr(rasa.studio.pull.pull, "read_assistant_name", read_from_root)
    monkeypatch.setattr(rasa.studio.push, "read_assistant_name", read_from_root)

    # Mock the StudioConfig and is_auth_working to simulate Studio connection
    monkeypatch.setattr(
        rasa.studio.link,
        "StudioConfig",
        MagicMock(
            read_config=lambda: StudioConfig(
                authentication_server_url="http://auth",
                studio_url="http://studio/graphql",
                realm_name="realm",
                client_id="cli",
            )
        ),
    )
    monkeypatch.setattr(rasa.studio.link, "is_auth_working", lambda *_: True)

    # Disable validation to avoid unnecessary checks during tests
    monkeypatch.setattr(rasa.studio.push, "run_validation", lambda *_: None)
    return tmp_path


@pytest.fixture
def mock_args(tmp_path: Path) -> MagicMock:
    args = MagicMock()
    args.domain = None
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir(parents=True, exist_ok=True)
    args.data = str(data_dir)
    args.overwrite = False
    args.config = None
    args.endpoints = None
    args.assistant_name = "my_assistant"
    return args


@pytest.fixture
def mock_studio_handler(
    system_prompts: Dict[Text, Text], monkeypatch: pytest.MonkeyPatch
) -> MagicMock:
    """
    Provide a fully-stubbed StudioDataHandler."""
    handler = MagicMock(spec=StudioDataHandler)

    handler.request_all_data.return_value = None
    handler.get_config = MagicMock()
    handler.get_config.return_value = "language: en\npipeline: []\n"
    handler.get_endpoints = MagicMock()
    handler.get_endpoints.return_value = (
        "action_endpoint:\n  url: http://localhost:5055\n"
    )
    handler.domain = textwrap.dedent(
        """
        version: '3.0'
        responses:
          utter_greet:
          - text: Hello!
        """
    ).lstrip()

    handler.has_nlu.return_value = False

    handler.has_flows.return_value = True
    handler.flows = textwrap.dedent(
        """
        flows:
          add_contact:
            name: Add a Contact
            description: Flow to add a contact to your contact list
            steps:
              - action: utter_add_contact
                next: END
          list_contacts:
            name: "list your contacts"
            description: "show your contact list"
            steps:
             - action: list_contacts
             - action: utter_no_contacts
               next: END
        """
    ).lstrip()

    handler.get_prompts = MagicMock()
    handler.get_prompts.return_value = system_prompts

    monkeypatch.setattr(
        "rasa.studio.download.StudioDataHandler",
        MagicMock(return_value=handler),
    )

    monkeypatch.setattr(
        "rasa.studio.download.questionary.confirm",
        MagicMock(return_value=MagicMock(ask=lambda: True)),
    )

    return handler
