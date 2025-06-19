import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import questionary
from pytest import MonkeyPatch

import rasa.studio.download
import rasa.studio.pull.pull
from rasa.shared.constants import DEFAULT_DATA_PATH
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import (
    DEFAULT_TRAINING_DATA_OUTPUT_PATH,
)
from rasa.studio.config import StudioConfig
from rasa.studio.constants import (
    DOMAIN_FILENAME,
)
from rasa.studio.data_handler import StudioDataHandler


def mock_questionary_confirm(question):
    return MagicMock(ask=lambda: "y")


@pytest.fixture
def test_sample_nlu() -> str:
    return """version: "3.1"
nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - hey there
    - let's go
    - hey dude
    - goodmorning
    - goodevening
    - good afternoon
- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later
- intent: new_intent
  examples: |
    - new example
    - new new example
- intent: inform
  examples: |
    - my name is [Uros](first_name)
    - I'm [John](first_name)
    - Hi, my first name is [Luis](first_name)
    - Karin
    - Steven
    - I'm [18](age)
    - I am [32](age) years old
    - 9(age)
- synonym: free-wifi
  examples: |
    - wifi
- synonym: LAST
  examples: |-
    - last one"""


@pytest.fixture
def test_sample_domain_nlu_only() -> str:
    return """version: "3.1"
intents:
  - greet
  - goodbye
  - inform
  - random_one
  - new_intent

entities:
  - first_name
  - new_entity
  - age
"""


@pytest.fixture
def test_sample_domain() -> str:
    return """version: "3.1"
intents:
  - greet
  - goodbye
  - inform
  - random_one
  - new_intent

entities:
  - first_name
  - new_entity
  - age

slots:
  logged_in:
    type: bool
    mappings:
    - type: from_text
  order_status:
    type: text
    mappings:
    - type: from_text

actions:
  - action_get_order_status
  - action_reset_unk_slots
  - validate_order_tracking_form

responses:
  utter_greet:
  - text: Hey! How are you?
  - text: Hey, {name}. Welcome back! How can I help you today?
    condition:
    - type: slot
      name: logged_in
      value: true
  utter_cheer_up:
  - text: 'Here is something to cheer you up:'
    image: https://i.imgur.com/nGF1K8f.jpg"""


@pytest.fixture
def test_sample_flows() -> str:
    return """flows:
  check_balance:
    name: check your balance
    description: check the user's account balance.

    steps:
      - action: check_balance
      - action: utter_current_balance
  replace_card:
    description: The user needs to replace their card.
    name: replace_card
    steps:
      - collect: confirm_correct_card
        ask_before_filling: true
        next:
          - if: "confirm_correct_card"
            then:
              - link: "replace_eligible_card"
          - else:
              - action: utter_relevant_card_not_linked
                next: END
  replace_eligible_card:
    description: replace eligible card
    name: replace_eligible_card
    steps:
      - action: utter_replacing_eligible_card
        next: END"""


def test_download_handler_nlu_based(
    tmp_path: Path,
    test_sample_nlu: str,
    test_sample_domain_nlu_only: str,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assistant_name = "test"
    name_space = argparse.Namespace(assistant_name=assistant_name)

    handler = StudioDataHandler(
        StudioConfig(
            authentication_server_url="http://studio.rasa",
            studio_url="http://studio.rasa",
            realm_name="rasa-test",
            client_id="rasa-cli",
        ),
        name_space.assistant_name,
    )
    handler.nlu = test_sample_nlu
    handler.domain = test_sample_domain_nlu_only
    handler.get_config = MagicMock(return_value="dummy config content")
    handler.get_endpoints = MagicMock(return_value="dummy endpoints content")
    monkeypatch.setattr(questionary, "confirm", mock_questionary_confirm)
    handler.request_all_data = MagicMock()  # type: ignore[method-assign]

    mock_handler = MagicMock()
    mock_handler.return_value = handler
    monkeypatch.setattr(rasa.studio.download, "StudioDataHandler", mock_handler)

    rasa.studio.download.handle_download(name_space)
    domain_path = tmp_path / assistant_name / DOMAIN_FILENAME
    assert domain_path.exists()
    training_data_path = (
        tmp_path
        / assistant_name
        / DEFAULT_DATA_PATH
        / DEFAULT_TRAINING_DATA_OUTPUT_PATH
    )
    assert training_data_path.exists()

    importer = TrainingDataImporter.load_from_dict(
        domain_path=domain_path,
        training_data_paths=[training_data_path],
    )

    domain = importer.get_domain()
    for intent in ["greet", "goodbye", "inform", "random_one", "new_intent"]:
        assert intent in domain.intents
    for entity in ["first_name", "new_entity", "age"]:
        assert entity in domain.entities

    nlu = importer.get_nlu_data()
    for intent in ["greet", "goodbye", "inform", "new_intent"]:
        assert intent in nlu.intents
    for entity in ["first_name", "age"]:
        assert entity in nlu.entities
    for synonym in ["last one", "wifi"]:
        assert synonym in nlu.entity_synonyms


def test_download_handler_modern(
    tmp_path: Path,
    test_sample_flows: str,
    test_sample_domain: str,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    assistant_name = "test"
    name_space = argparse.Namespace(assistant_name=assistant_name)
    handler = StudioDataHandler(
        StudioConfig(
            authentication_server_url="http://studio.amazonaws.com",
            studio_url="http://studio.amazonaws.com",
            realm_name="rasa-test",
            client_id="rasa-cli",
        ),
        name_space.assistant_name,
    )
    handler.flows = test_sample_flows
    handler.domain = test_sample_domain
    handler.get_config = MagicMock(return_value="dummy config content")
    handler.get_endpoints = MagicMock(return_value="dummy endpoints content")
    handler.request_all_data = MagicMock()  # type: ignore[method-assign]
    mock_handler = MagicMock()
    mock_handler.return_value = handler
    monkeypatch.setattr(rasa.studio.download, "StudioDataHandler", mock_handler)
    monkeypatch.setattr(questionary, "confirm", mock_questionary_confirm)
    rasa.studio.download.handle_download(name_space)

    domain_path = tmp_path / assistant_name / DOMAIN_FILENAME
    assert domain_path.exists()
    training_data_path = tmp_path / assistant_name / DEFAULT_DATA_PATH
    assert training_data_path.exists()
    importer = TrainingDataImporter.load_from_dict(
        domain_path=domain_path,
        training_data_paths=[training_data_path],
    )

    domain = importer.get_domain()
    for slot_name in ["logged_in", "order_status"]:
        assert slot_name in [slot.name for slot in domain.slots]
    for action_name in [
        "action_get_order_status",
        "action_reset_unk_slots",
        "validate_order_tracking_form",
    ]:
        assert action_name in domain.action_names_or_texts
    for response_name in ["utter_greet", "utter_cheer_up"]:
        assert response_name in domain.responses

    flows = importer.get_flows().underlying_flows
    for flow_name in ["check_balance", "replace_card"]:
        assert flow_name in [flow.id for flow in flows]

    for intent in ["greet", "goodbye", "inform", "random_one", "new_intent"]:
        assert intent in domain.intents
    for entity in ["first_name", "new_entity", "age"]:
        assert entity in domain.entities
