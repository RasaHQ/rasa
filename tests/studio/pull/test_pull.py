import textwrap
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import questionary

import rasa.studio.link
import rasa.studio.pull.pull
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import Flow
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import read_yaml
from rasa.studio.constants import STUDIO_DOMAIN_FILENAME


def test_handle_pull_config(project: Path, monkeypatch: pytest.MonkeyPatch):
    handler = MagicMock()
    handler.get_config.return_value = "language: en\npipeline: []"
    monkeypatch.setattr(
        rasa.studio.pull.pull, "StudioDataHandler", lambda *_, **__: handler
    )

    args = Namespace(config="config.yml")
    rasa.studio.pull.pull.handle_pull_config(args)

    written = (project / "config.yml").read_text()
    assert written.startswith("language: en")


def test_handle_pull_endpoints(project: Path, monkeypatch: pytest.MonkeyPatch):
    handler = MagicMock()
    handler.get_endpoints.return_value = "nlg:\n  type: utter"
    monkeypatch.setattr(
        rasa.studio.pull.pull, "StudioDataHandler", lambda *_, **__: handler
    )

    args = Namespace(endpoints="endpoints.yml")
    rasa.studio.pull.pull.handle_pull_endpoints(args)

    written = (project / "endpoints.yml").read_text()
    assert "type: utter" in written


def test_pull_all_creates_files(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assistant_path = tmp_path / "my_assistant"
    assistant_path.mkdir()
    monkeypatch.chdir(assistant_path)

    handler = MagicMock()
    handler.has_nlu.return_value = False
    handler.get_config.return_value = "some config text"
    handler.get_endpoints.return_value = "some endpoints text"
    handler.domain = "some domain text"
    monkeypatch.setattr(
        rasa.studio.pull.pull, "StudioDataHandler", lambda *_, **__: handler
    )

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_from_studio.get_user_domain.return_value = Domain.from_dict(
        {
            "responses": {
                "some_response": [
                    {
                        "text": "some text",
                    }
                ]
            }
        }
    )
    data_from_studio.get_user_flows.return_value = FlowsList(
        [
            Flow.from_json(
                "foo",
                {
                    "description": "a test flow",
                    "steps": [{"id": "first", "action": "action_listen"}],
                },
            )
        ]
    )
    data_local = MagicMock(spec=TrainingDataImporter)
    mock_import_data = MagicMock(return_value=(data_from_studio, data_local))
    monkeypatch.setattr(
        "rasa.studio.pull.pull.import_data_from_studio", mock_import_data
    )
    monkeypatch.setattr(questionary, "confirm", MagicMock(ask=lambda: "y"))

    args = Namespace(
        domain="domain.yml",
        data="data",
        config="config.yml",
        endpoints="endpoints.yml",
    )
    rasa.studio.pull.pull.handle_pull(args)

    assert "some config text" in (assistant_path / "config.yml").read_text()
    assert "some endpoints text" in (assistant_path / "endpoints.yml").read_text()
    assert (
        "some_response:\n  - text: some text"
        in (assistant_path / STUDIO_DOMAIN_FILENAME).read_text()
    )
    assert (
        "description: a test flow"
        in (assistant_path / "data" / "flows" / "foo.yml").read_text()
    )


def test_handle_pull_overwrites_domain(
    project: Path,
    mock_args: MagicMock,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test pull studio data with overwriting domain."""
    # Local domain contains `name` entity with `Jane` value
    domain_file = project / "domain.yml"
    entity = {"entity": "name", "value": "Jane", "role": "contact", "group": "test"}
    domain = Domain.from_dict({"entities": [entity]})
    domain_file.write_text(domain.as_yaml())
    data_local = MagicMock(spec=TrainingDataImporter)
    data_local.get_user_domain.return_value = domain

    # Studio domain contains `name` entity with `John` value
    entity["value"] = "John"
    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_from_studio.get_user_domain.return_value = Domain.from_dict(
        {"entities": [entity]}
    )

    monkeypatch.setattr(
        "rasa.studio.pull.pull.import_data_from_studio",
        MagicMock(return_value=(data_from_studio, data_local)),
    )
    monkeypatch.setattr(
        "rasa.studio.pull.pull._create_studio_handler",
        lambda *args, **kwargs: mock_studio_handler,
    )
    mock_args.domain = domain_file
    rasa.studio.pull.pull.handle_pull(mock_args)

    # Leftover domain should not exist
    leftover_path = project / STUDIO_DOMAIN_FILENAME
    assert not leftover_path.exists()

    # Local domain should have the entity updated with the value from Studio
    updated_domain = Domain.from_file(str(domain_file))
    assert updated_domain.as_dict()["entities"][0]["value"] == "John"


def test_handle_pull_overwrites_flows(
    project: Path,
    mock_args: MagicMock,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test pull studio data with overwriting flows."""
    data_local = MagicMock(spec=TrainingDataImporter)

    add_contact_flow_yaml = textwrap.dedent(
        """
          add_contact:
            name: Add a Contact Edited
            description: Flow to add a contact to your contact list
            steps:
              - action: utter_add_contact
                next: END
        """
    )
    studio_flows = FlowsList.from_json(read_yaml(add_contact_flow_yaml))
    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_from_studio.get_user_flows.return_value = studio_flows

    monkeypatch.setattr(
        "rasa.studio.pull.pull.import_data_from_studio",
        MagicMock(return_value=(data_from_studio, data_local)),
    )
    monkeypatch.setattr(
        "rasa.studio.pull.pull._create_studio_handler",
        lambda *a, **kw: mock_studio_handler,
    )

    mock_args.data = "data"
    monkeypatch.chdir(project)

    rasa.studio.pull.pull.handle_pull(mock_args)

    local_flow_path = project / "data" / "flows" / "add_contact.yml"
    resulting_content = local_flow_path.read_text()
    flow_yaml = read_yaml(resulting_content)
    assert (
        flow_yaml["flows"]["add_contact"]["name"]
        == studio_flows.underlying_flows[0].name
    )
