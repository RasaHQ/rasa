import os
from pathlib import Path
from typing import Any, Dict, List, Text, Type
from unittest.mock import MagicMock

import pytest
import structlog
from pytest import MonkeyPatch

import rasa.shared.core.constants
import rasa.shared.utils.io
from rasa.shared.constants import (
    CONFIG_ADDITIONAL_LANGUAGES_KEY,
    CONFIG_LANGUAGE_KEY,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONVERSATION_TEST_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, SlotSet, UserUttered
from rasa.shared.core.training_data.structures import StoryGraph, StoryStep
from rasa.shared.exceptions import RasaException
from rasa.shared.importers.importer import (
    E2EImporter,
    LanguageImporter,
    NluDataImporter,
    ResponsesSyncImporter,
    TrainingDataImporter,
)
from rasa.shared.importers.multi_project import MultiProjectImporter
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.constants import ACTION_NAME, ACTION_TEXT, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.yaml import write_yaml
from tests.utilities import filter_logs


@pytest.fixture()
def default_importer(project: Text) -> TrainingDataImporter:
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    return TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, [default_data_path]
    )


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, [RasaFileImporter]),
        ({"importers": []}, [RasaFileImporter]),
        ({"importers": [{"name": "RasaFileImporter"}]}, [RasaFileImporter]),
        ({"importers": [{"name": "NotExistingModule"}]}, [RasaFileImporter]),
        (
            {
                "importers": [
                    {"name": "rasa.shared.importers.multi_project.MultiProjectImporter"}
                ]
            },
            [MultiProjectImporter],
        ),
        ({"importers": [{"name": "MultiProjectImporter"}]}, [MultiProjectImporter]),
        (
            {
                "importers": [
                    {"name": "RasaFileImporter"},
                    {"name": "MultiProjectImporter"},
                ]
            },
            [RasaFileImporter, MultiProjectImporter],
        ),
    ],
)
def test_load_from_dict(
    config: Dict, expected: List[Type["TrainingDataImporter"]], project: Text
):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    actual = TrainingDataImporter.load_from_dict(
        config, config_path, domain_path, [default_data_path]
    )

    assert isinstance(actual, LanguageImporter)
    assert isinstance(actual._importer._importer._importer, ResponsesSyncImporter)

    actual_importers = [
        i.__class__ for i in actual._importer._importer._importer._importer._importers
    ]
    assert actual_importers == expected


def test_load_from_config(tmpdir: Path):
    config_path = str(tmpdir / "config.yml")

    write_yaml({"importers": [{"name": "MultiProjectImporter"}]}, config_path)

    importer = TrainingDataImporter.load_from_config(config_path)
    assert isinstance(importer, LanguageImporter)
    assert isinstance(importer._importer._importer._importer, ResponsesSyncImporter)
    assert isinstance(
        importer._importer._importer._importer._importer._importers[0],
        MultiProjectImporter,
    )


def test_nlu_only(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    actual = TrainingDataImporter.load_nlu_importer_from_config(
        config_path, training_data_paths=[default_data_path]
    )

    assert isinstance(actual, NluDataImporter)
    assert isinstance(
        actual._importer._importer._importer._importer, ResponsesSyncImporter
    )

    stories = actual.get_stories()
    assert stories.is_empty()

    conversation_tests = actual.get_stories()
    assert conversation_tests.is_empty()

    domain = actual.get_domain()
    assert domain.is_empty()

    config = actual.get_config()
    assert config

    nlu_data = actual.get_nlu_data()
    assert not nlu_data.is_empty()


def test_import_nlu_training_data_from_e2e_stories(
    default_importer: TrainingDataImporter,
):
    # The `E2EImporter` correctly wraps the underlying `CombinedDataImporter`
    assert isinstance(default_importer, LanguageImporter)
    importer_without_e2e = default_importer._importer._importer

    stories = StoryGraph(
        [
            StoryStep(
                "name",
                events=[
                    SlotSet("some slot", "doesn't matter"),
                    UserUttered(intent={"name": "greet_from_stories"}),
                    ActionExecuted("utter_greet_from_stories"),
                ],
            ),
            StoryStep(
                "name",
                events=[
                    UserUttered("how are you doing?"),
                    ActionExecuted(action_text="Hi Joey."),
                ],
            ),
        ]
    )

    def mocked_stories(*_: Any, **__: Any) -> StoryGraph:
        return stories

    # Patch to return our test stories
    importer_without_e2e.get_stories = mocked_stories

    # The wrapping `E2EImporter` simply forwards these method calls
    assert (importer_without_e2e.get_stories()).fingerprint() == (
        default_importer.get_stories()
    ).fingerprint()
    assert (importer_without_e2e.get_config()) == (default_importer.get_config())

    # Check additional NLU training data from stories was added
    nlu_data = default_importer.get_nlu_data()

    # The `E2EImporter` adds NLU training data based on our training stories
    assert len(nlu_data.training_examples) > len(
        importer_without_e2e.get_nlu_data().training_examples
    )

    # Check if the NLU training data was added correctly from the story training data
    expected_additional_messages = [
        Message(data={INTENT: "greet_from_stories"}),
        Message(data={ACTION_NAME: "utter_greet_from_stories"}),
        Message(data={TEXT: "how are you doing?"}),
        Message(data={ACTION_TEXT: "Hi Joey."}),
    ]

    assert all(m in nlu_data.training_examples for m in expected_additional_messages)


def test_different_story_order_doesnt_change_nlu_training_data(
    default_importer: E2EImporter,
):
    stories = [
        StoryStep(
            "name",
            events=[
                UserUttered(intent={"name": "greet"}),
                ActionExecuted("utter_greet_from_stories"),
                ActionExecuted("hi", action_text="hi"),
            ],
        ),
        StoryStep(
            "name",
            events=[
                UserUttered("bye", {"name": "bye"}),
                ActionExecuted("utter_greet"),
                ActionExecuted("hi", action_text="hi"),
                ActionExecuted("bye", action_text="bye"),
            ],
        ),
    ]

    def mocked_stories(*_: Any, **__: Any) -> StoryGraph:
        return StoryGraph(stories)

    # Patch to return our test stories
    default_importer._importer.get_stories = mocked_stories

    training_data = default_importer.get_nlu_data()

    # Pretend the order of  the stories changed. This should have no
    # effect on the NLU training data
    stories = list(reversed(stories))

    # Make sure importer doesn't cache stories
    default_importer._cached_stories = None

    training_data2 = default_importer.get_nlu_data()

    assert hash(training_data) == hash(training_data2)


def test_import_nlu_training_data_with_default_actions(
    default_importer: TrainingDataImporter,
):
    assert isinstance(default_importer, LanguageImporter)
    importer_without_e2e = default_importer._importer._importer

    # Check additional NLU training data from domain was added
    nlu_data = default_importer.get_nlu_data()

    assert len(nlu_data.training_examples) > len(
        importer_without_e2e.get_nlu_data().training_examples
    )

    extended_training_data = default_importer.get_nlu_data()
    assert all(
        Message(data={ACTION_NAME: action_name})
        in extended_training_data.training_examples
        for action_name in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
    )


def test_adding_e2e_actions_to_domain(default_importer: E2EImporter):
    additional_actions = ["Hi Joey.", "it's sunny outside."]
    stories = StoryGraph(
        [
            StoryStep(
                "name",
                events=[
                    UserUttered("greet_from_stories", {"name": "greet_from_stories"}),
                    ActionExecuted("utter_greet_from_stories"),
                ],
            ),
            StoryStep(
                "name",
                events=[
                    UserUttered("how are you doing?", {"name": "greet_from_stories"}),
                    ActionExecuted(
                        additional_actions[0], action_text=additional_actions[0]
                    ),
                    ActionExecuted(
                        additional_actions[1], action_text=additional_actions[1]
                    ),
                    ActionExecuted(
                        additional_actions[1], action_text=additional_actions[1]
                    ),
                ],
            ),
        ]
    )

    def mocked_stories(*_: Any, **__: Any) -> StoryGraph:
        return stories

    # Patch to return our test stories
    default_importer._importer.get_stories = mocked_stories

    domain = default_importer.get_domain()

    assert all(
        action_name in domain.action_names_or_texts
        for action_name in additional_actions
    )


def test_nlu_data_domain_sync_with_retrieval_intents(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = "data/test_domains/default_retrieval_intents.yml"
    data_paths = [
        "data/test/stories_default_retrieval_intents.yml",
        "data/test_responses/default.yml",
    ]
    importer = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, data_paths
    )

    domain = importer.get_domain()
    nlu_data = importer.get_nlu_data()

    assert domain.retrieval_intents == ["chitchat"]
    assert domain.intent_properties["chitchat"].get("is_retrieval_intent")
    assert domain.retrieval_intent_responses == nlu_data.responses
    assert domain.responses != nlu_data.responses
    assert "utter_chitchat" in domain.action_names_or_texts


def test_subintent_response_matches_with_action(project: Text):
    """Tests retrieval intent responses are matched correctly to actions."""
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = "data/test_domains/simple_retrieval_intent.yml"
    data_path = "data/test/simple_retrieval_intent_nlu.yml"
    importer = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, data_path
    )

    domain = importer.get_domain()
    # Test retrieval intent response is matched correctly to actions
    # ie. utter_chitchat/faq response compatible with action utter_chitchat
    with pytest.warns(None) as record:
        domain.check_missing_responses()
    assert not record


def test_response_missing(project: Text):
    """Tests warning when response is missing."""
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = "data/test_domains/missing_chitchat_response.yml"
    data_path = "data/test/simple_retrieval_intent_nlu.yml"
    importer = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, data_path
    )

    domain = importer.get_domain()

    expected_log_level = "warning"
    expected_log_event = "domain.check_missing_response"
    expected_log_message = (
        "Action 'utter_chitchat' is listed as a response action in the domain "
        "file, but there is no matching response defined. Please check your "
        "domain."
    )

    with structlog.testing.capture_logs() as caplog:
        domain.check_missing_responses()
        logs = filter_logs(
            caplog, expected_log_event, expected_log_level, [expected_log_message]
        )
        assert len(logs) == 1


def test_nlu_data_domain_sync_responses(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = "data/test_domains/default.yml"
    data_paths = ["data/test_responses/responses_utter_rasa.yml"]

    importer = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, data_paths
    )

    with pytest.warns(None):
        domain = importer.get_domain()

    # Responses were sync between "test_responses.yml" and the "domain.yml"
    assert "utter_rasa" in domain.responses.keys()


def test_importer_with_unicode_files():
    importer = TrainingDataImporter.load_from_dict(
        training_data_paths=["./data/test_nlu_no_responses/nlu_with_unicode.yml"]
    )

    # None of these should raise
    nlu_data = importer.get_nlu_data()
    assert not nlu_data.is_empty()

    importer.get_stories()
    importer.get_domain()


def test_read_conversation_tests(project: Text):
    importer = TrainingDataImporter.load_from_dict(
        training_data_paths=[str(Path(project) / DEFAULT_CONVERSATION_TEST_PATH)]
    )

    test_stories = importer.get_conversation_tests()
    assert len(test_stories.story_steps) == 7


def test_importer_fingerprint():
    importer = TrainingDataImporter.load_from_dict(
        training_data_paths=["./data/test_nlu_no_responses/nlu_with_unicode.yml"]
    )

    fp1 = importer.fingerprint()
    fp2 = importer.fingerprint()
    assert fp1 != fp2


def test_language_importer_adds_language_slot(
    default_importer: TrainingDataImporter, monkeypatch: MonkeyPatch
):
    # Mock languages in config.yml
    config = {CONFIG_LANGUAGE_KEY: "de", CONFIG_ADDITIONAL_LANGUAGES_KEY: ["en"]}
    monkeypatch.setattr(
        "rasa.shared.importers.importer.PassThroughImporter.get_config",
        MagicMock(return_value=config),
    )

    # Initialize LanguageImporter with default importer
    language_importer = LanguageImporter(default_importer)

    # Verify the language slot is added to the domain
    domain = language_importer.get_domain()
    language_slot_name = rasa.shared.core.constants.LANGUAGE_SLOT
    slots_map = {slot.name: slot for slot in domain.slots}
    assert language_slot_name in slots_map

    # Verify the slot's values list includes the language and additional languages
    language_slot = slots_map[language_slot_name]
    assert language_slot.initial_value == "de"
    assert language_slot.values == ["en", "de"]


def test_language_importer_adds_language_slot_without_additional_languages(
    default_importer: TrainingDataImporter, monkeypatch: MonkeyPatch
):
    # Mock languages in config.yml
    config = {CONFIG_LANGUAGE_KEY: "de"}
    monkeypatch.setattr(
        "rasa.shared.importers.importer.PassThroughImporter.get_config",
        MagicMock(return_value=config),
    )

    # Initialize LanguageImporter with default importer
    language_importer = LanguageImporter(default_importer)

    # Verify the language slot is added to the domain
    domain = language_importer.get_domain()
    language_slot_name = rasa.shared.core.constants.LANGUAGE_SLOT
    slots_map = {slot.name: slot for slot in domain.slots}
    assert language_slot_name in slots_map

    # Verify the slot's values list includes the language and additional languages
    language_slot = slots_map[language_slot_name]
    assert language_slot.initial_value == "de"
    assert language_slot.values == ["de"]


def test_builtin_language_slot_overriden(
    default_importer: TrainingDataImporter, monkeypatch: MonkeyPatch
):
    domain = Domain.from_yaml(
        """
        slots:
            language:
                type: strict_categorical
                initial_value: en
                values:
                    - en
                    - de
                    - it
        """
    )
    monkeypatch.setattr(default_importer, "get_domain", MagicMock(return_value=domain))

    # Initialize LanguageImporter with default importer
    language_importer = LanguageImporter(default_importer)
    with pytest.raises(RasaException) as exc_info:
        language_importer.get_domain()

    expected = "The 'language' slot is a builtin slot that cannot be overridden."
    assert expected in str(exc_info.value)
