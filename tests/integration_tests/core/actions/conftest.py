from typing import Text
import copy
import pytest

from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture(scope="session")
def domain_yaml_content() -> Text:
    return """
intents:
 - greet
 - default
 - goodbye
 - affirm
 - thank_you
 - change_bank_details
 - simple
 - hello
 - why
 - next_intent

entities:
 - name

slots:
  name:
    type: text
    mappings:
      - type: from_entity
        entity: name

responses:
  utter_greet:
    - text: "hey there {name}!"
  utter_channel:
    - text: "this is a default channel"
    - text: "you're talking to me on slack!"
      channel: "slack"
  utter_goodbye:
    - text: "goodbye 😢"
    - text: "bye bye 😢"
  utter_default:
    - text: "sorry, I didn't get that, can you rephrase it?"

forms:
  some_form:
    required_slots:
      - name
"""


@pytest.fixture(scope="session")
def _domain(domain_yaml_content: Text) -> Domain:
    return Domain.from_yaml(domain_yaml_content)


@pytest.fixture()
def domain(_domain: Domain) -> Domain:
    return copy.deepcopy(_domain)


@pytest.fixture
def default_tracker(domain: Domain) -> DialogueStateTracker:
    return DialogueStateTracker("my-sender", domain.slots)


@pytest.fixture
def domain_with_response_ids() -> Domain:
    domain_yaml = """
    responses:
        utter_one_id:
            - text: test
              id: '1'
        utter_multiple_ids:
            - text: test
              id: '2'
            - text: test
              id: '3'
        utter_no_id:
            - text: test
    """
    domain = Domain.from_yaml(domain_yaml)
    return domain
