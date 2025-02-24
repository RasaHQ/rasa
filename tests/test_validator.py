import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Text, Union
from unittest.mock import MagicMock, patch

import pytest
import structlog
from pytest import CaptureFixture, MonkeyPatch

from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.telemetry import (
    TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE,
    TELEMETRY_VALIDATION_ERROR_LOG_EVENT,
)
from rasa.validator import Validator
from tests.utilities import filter_logs, flows_from_str


@pytest.fixture(scope="class")
def validator_under_test() -> Validator:
    importer = RasaFileImporter(
        domain_path="data/test_validation/domain.yml",
        training_data_paths=[
            "data/test_validation/data/nlu.yml",
            "data/test_validation/data/stories.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    return validator


def test_verify_nlu_with_e2e_story(
    tmp_path: Path, nlu_data_path: Path, capsys: CaptureFixture
):
    story_file_name = tmp_path / "stories.yml"
    with open(story_file_name, "w") as file:
        file.write(
            """
            stories:
            - story: path 1
              steps:
              - user: |
                  hello assistant! Can you help me today?
              - intent: greet
              - action: utter_greet
              - intent: affirm
              - action: utter_greet
              - intent: bot_challenge
              - action: utter_greet
              - intent: deny
              - action: goodbye
              - intent: goodbye
              - action: utter_goodbye
              - intent: mood_great
              - action: utter_happy
              - intent: mood_unhappy
              - action: utter_cheer_up
              - action: utter_did_that_help
              - action: utter_iamabot
            """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path="data/test_moodbot/domain.yml",
        training_data_paths=[story_file_name, nlu_data_path],
    )

    expected_event = (
        "validator.verify_example_repetition_in_intents" ".one_example_multiple_intents"
    )
    expected_log_level = "warning"
    expected_log_message = (
        "The example 'good afternoon' was found labeled " "with multiple different"
    )

    validator = Validator.from_importer(importer)
    # Since the nlu file actually fails validation,
    # record warnings to make sure that the only raised warning
    # is about the duplicate example 'good afternoon'
    validator.verify_nlu(ignore_warnings=False)

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_intents_does_not_fail_on_valid_data(nlu_data_path: Text):
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path]
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_intents(ignore_warnings=False)


def test_verify_intents_does_fail_on_invalid_data(nlu_data_path: Text):
    # domain and nlu data are from different domain and should produce warnings
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml", training_data_paths=[nlu_data_path]
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert not validator.verify_intents(ignore_warnings=False)


def test_verify_story_structure(stories_path: Text):
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml", training_data_paths=[stories_path]
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_story_structure(ignore_warnings=False)


def test_verify_bad_story_structure():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_yaml_stories/stories_conflicting_2.yml"],
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert not validator.verify_story_structure(ignore_warnings=False)


def test_verify_bad_e2e_story_structure_when_text_identical(tmp_path: Path):
    story_file_name = tmp_path / "stories.yml"
    story_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: path 1
          steps:
          - user: |
              amazing!
          - action: utter_happy
        - story: path 2 (should always conflict path 1)
          steps:
          - user: |
              amazing!
          - action: utter_cheer_up
        """
    )
    # The two stories with identical user texts
    importer = RasaFileImporter(
        config_file="data/test_config/config_defaults.yml",
        domain_path="data/test_domains/default.yml",
        training_data_paths=[story_file_name],
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert not validator.verify_story_structure(ignore_warnings=False)


def test_verify_correct_e2e_story_structure(tmp_path: Path):
    story_file_name = tmp_path / "stories.yml"
    with open(story_file_name, "w") as file:
        file.write(
            """
            stories:
            - story: path 1
              steps:
              - user: |
                  hello assistant! Can you help me today?
              - action: utter_greet
            - story: path 2 - state is similar but different from the one in path 1
              steps:
              - user: |
                  hello assistant! you Can help me today?
              - action: utter_goodbye
            - story: path 3
              steps:
              - user: |
                  That's it for today. Chat again tomorrow!
              - action: utter_goodbye
            """
        )
    importer = RasaFileImporter(
        config_file="data/test_config/config_defaults.yml",
        domain_path="data/test_domains/default.yml",
        training_data_paths=[story_file_name],
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_story_structure(ignore_warnings=False)


def test_verify_correct_e2e_story_structure_with_intents(tmp_path: Path):
    story_file_name = tmp_path / "stories.yml"
    with open(story_file_name, "w") as file:
        file.write(
            """
            stories:
            - story: path 1
              steps:
              - intent: greet
              - action: utter_greet
            - story: path 2
              steps:
              - intent: goodbye
              - action: utter_goodbye
            """
        )
    importer = RasaFileImporter(
        config_file="data/test_config/config_defaults.yml",
        domain_path="data/test_domains/default.yml",
        training_data_paths=[story_file_name],
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_story_structure(ignore_warnings=False)


def test_verify_story_structure_ignores_rules():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=[
            "data/test_yaml_stories/stories_with_rules_conflicting.yml"
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_story_structure(ignore_warnings=False)


def test_verify_bad_story_structure_ignore_warnings():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_yaml_stories/stories_conflicting_2.yml"],
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_story_structure(ignore_warnings=True)


def test_verify_there_is_example_repetition_in_intents(nlu_data_path: Text):
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye

    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path]
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert not validator.verify_example_repetition_in_intents(ignore_warnings=False)


def test_verify_logging_message_for_intent_not_used_in_nlu(
    validator_under_test: Validator,
    capsys: CaptureFixture,
):
    expected_event = "validator.verify_intents.not_in_nlu_training_data"
    expected_log_level = "warning"
    expected_log_message = (
        "The intent 'goodbye' is listed in the domain "
        "file, but is not found in the NLU training data."
    )

    # force validator to not ignore warnings (default is True)
    validator_under_test.verify_intents(ignore_warnings=False)

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_logging_message_for_intent_not_used_in_story(
    validator_under_test: Validator,
    capsys: CaptureFixture,
):
    expected_event = "validator.verify_intents_in_stories_or_flows.not_used"
    expected_log_level = "warning"
    expected_log_message = (
        "The intent 'goodbye' is not used " "in any story, rule or flow."
    )

    validator_under_test.verify_intents_in_stories_or_flows(ignore_warnings=False)

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_logging_message_for_repetition_in_intents(
    nlu_data_path: Text, capsys: CaptureFixture
):
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path]
    )
    validator = Validator.from_importer(importer)

    expected_event = (
        "validator.verify_example_repetition_in_intents" ".one_example_multiple_intents"
    )
    expected_log_level = "warning"
    expected_log_message_part = "You should fix that conflict "

    validator.verify_example_repetition_in_intents(ignore_warnings=False)

    result = capsys.readouterr()
    assert expected_log_message_part in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_early_exit_on_invalid_domain():
    domain_path = "data/test_domains/duplicate_intents.yml"

    importer = RasaFileImporter(domain_path=domain_path)
    with pytest.warns(UserWarning) as record:
        warnings.simplefilter("ignore", DeprecationWarning)
        validator = Validator.from_importer(importer)
    validator.verify_domain_validity()

    # one for non-unique domain
    assert len(record) == 1

    non_unique_warnings = list(
        filter(
            lambda warning: f"Loading domain from '{domain_path}' failed. "
            f"Using empty domain. Error: 'Intents are not unique! "
            f"Found multiple intents with name(s) ['default', 'goodbye']. "
            f"Either rename or remove the duplicate ones.'" in warning.message.args[0],
            record,
        )
    )
    assert len(non_unique_warnings) == 1


def test_verify_there_is_not_example_repetition_in_intents():
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml",
        training_data_paths=["examples/nlu_based/knowledgebasebot/data/nlu.yml"],
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_example_repetition_in_intents(ignore_warnings=False)


def test_verify_actions_in_stories_not_in_domain(
    tmp_path: Path, domain_path: Text, capsys: CaptureFixture
):
    story_file_name = tmp_path / "stories.yml"
    story_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: story path 1
          steps:
          - intent: greet
          - action: action_test_1
        """
    )

    importer = RasaFileImporter(
        domain_path=domain_path, training_data_paths=[story_file_name]
    )
    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_actions_in_stories_rules.not_in_domain"
    expected_log_level = "error"
    expected_log_message = (
        "The action 'action_test_1' is used in "
        "the 'story path 1' block, but it is "
        "not listed in the domain file."
    )

    assert not validator.verify_actions_in_stories_rules()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_actions_in_rules_not_in_domain(
    tmp_path: Path, domain_path: Text, capsys: CaptureFixture
):
    rules_file_name = tmp_path / "rules.yml"
    rules_file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        rules:
        - rule: rule path 1
          steps:
          - intent: goodbye
          - action: action_test_2
        """
    )
    importer = RasaFileImporter(
        domain_path=domain_path, training_data_paths=[rules_file_name]
    )
    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_actions_in_stories_rules.not_in_domain"
    expected_log_level = "error"
    expected_log_message = (
        "The action 'action_test_2' is used in the "
        "'rule path 1' block, but it is not listed in "
        "the domain file."
    )

    assert not validator.verify_actions_in_stories_rules()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_form_slots_invalid_domain(tmp_path: Path, capsys: CaptureFixture):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        forms:
          name_form:
            required_slots:
              - first_name
              - last_nam
        slots:
             first_name:
                type: text
                mappings:
                - type: from_text
             last_name:
                type: text
                mappings:
                - type: from_text
        """
    )
    importer = RasaFileImporter(domain_path=domain)
    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_form_slots.not_in_domain"
    expected_log_level = "warning"
    expected_log_message = (
        "The form slot 'last_nam' in form 'name_form' "
        "is not present in the domain slots.Please "
        "add the correct slot or check for typos."
    )

    assert not validator.verify_form_slots()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_invalid_domain_mapping_policy():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default_with_mapping.yml"
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_domain_validity() is False


@pytest.mark.parametrize(
    ("file_name", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_valid_stories_rules_actions_in_domain(
    file_name: Text, data_type: Text, tmp_path: Path
):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - greet
        actions:
        - action_greet
        """
    )
    file_name = tmp_path / f"{file_name}.yml"
    file_name.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        {file_name}:
        - {data_type}: test path
          steps:
          - intent: greet
          - action: action_greet
        """
    )
    importer = RasaFileImporter(domain_path=domain, training_data_paths=[file_name])
    validator = Validator.from_importer(importer)
    assert validator.verify_actions_in_stories_rules()


@pytest.mark.parametrize(
    ("file_name", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_valid_stories_rules_default_actions(
    file_name: Text, data_type: Text, tmp_path: Path
):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - greet
        """
    )
    file_name = tmp_path / f"{file_name}.yml"
    file_name.write_text(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            {file_name}:
            - {data_type}: test path
              steps:
              - intent: greet
              - action: action_restart
            """
    )
    importer = RasaFileImporter(domain_path=domain, training_data_paths=[file_name])
    validator = Validator.from_importer(importer)
    assert validator.verify_actions_in_stories_rules()


def test_valid_form_slots_in_domain(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        forms:
          name_form:
            required_slots:
              - first_name
              - last_name
        slots:
             first_name:
                type: text
                mappings:
                - type: from_text
             last_name:
                type: text
                mappings:
                - type: from_text
        """
    )
    importer = RasaFileImporter(domain_path=domain)
    validator = Validator.from_importer(importer)
    assert validator.verify_form_slots()


def test_verify_slot_mappings_mapping_active_loop_not_in_forms(
    tmp_path: Path, capsys: CaptureFixture
):
    domain = tmp_path / "domain.yml"
    slot_name = "some_slot"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        entities:
        - some_entity
        slots:
          {slot_name}:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: some_entity
              conditions:
              - active_loop: som_form
        forms:
          some_form:
            required_slots:
              - {slot_name}
        """
    )
    importer = RasaFileImporter(domain_path=domain)
    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_slot_mappings.not_in_domain"
    expected_log_level = "warning"
    expected_log_message = (
        "Slot 'some_slot' has a mapping condition "
        "for form 'som_form' which is not "
        "listed in domain forms."
    )

    assert not validator.verify_slot_mappings()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_slot_mappings_slot_with_mapping_conditions_not_in_form(
    tmp_path: Path, capsys: CaptureFixture
):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - activate_booking
        entities:
        - city
        slots:
          location:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: city
              conditions:
              - active_loop: booking_form
          started_booking_form:
            type: bool
            influence_conversation: false
            mappings:
            - type: from_trigger_intent
              intent: activate_booking
              value: true
        forms:
          booking_form:
            required_slots:
            - started_booking_form
            """
    )
    importer = RasaFileImporter(domain_path=domain)
    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_slot_mappings.not_in_forms_key"
    expected_log_level = "warning"
    expected_log_message = (
        "Slot 'location' has a mapping "
        "condition for form 'booking_form', "
        "but it's not present in 'booking_form' "
        "form's 'required_slots'."
    )

    assert not validator.verify_slot_mappings()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_slot_mappings_valid(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - activate_booking
        entities:
        - city
        slots:
          location:
            type: text
            influence_conversation: false
            mappings:
            - type: from_entity
              entity: city
              conditions:
              - active_loop: booking_form
          started_booking_form:
            type: bool
            influence_conversation: false
            mappings:
            - type: from_trigger_intent
              intent: activate_booking
              value: true
        forms:
          booking_form:
            required_slots:
            - started_booking_form
            - location
            """
    )
    importer = RasaFileImporter(domain_path=domain)
    validator = Validator.from_importer(importer)
    assert validator.verify_slot_mappings()


@pytest.mark.parametrize(
    ("file_name", "data_type"), [("stories", "story"), ("rules", "rule")]
)
def test_default_action_as_active_loop_in_rules(
    tmp_path: Path, file_name: Text, data_type: Text
) -> None:
    config = tmp_path / "config.yml"

    config.write_text(
        textwrap.dedent(
            """
            recipe: default.v1
            language: en
            pipeline:
               - name: WhitespaceTokenizer
               - name: RegexFeaturizer
               - name: LexicalSyntacticFeaturizer
               - name: CountVectorsFeaturizer
               - name: CountVectorsFeaturizer
                 analyzer: char_wb
                 min_ngram: 1
                 max_ngram: 4
               - name: DIETClassifier
                 epochs: 100
               - name: EntitySynonymMapper
               - name: ResponseSelector
                 epochs: 100
               - name: FallbackClassifier
                 threshold: 0.3
                 ambiguity_threshold: 0.1
            policies:
               - name: MemoizationPolicy
               - name: TEDPolicy
                 max_history: 5
                 epochs: 100
               - name: RulePolicy
                 core_fallback_threshold: 0.3
                 core_fallback_action_name: "action_default_fallback"
                 enable_fallback_prediction: true
            """
        )
    )

    domain = tmp_path / "domain.yml"
    domain.write_text(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
              - greet
              - goodbye
              - affirm
              - deny
              - mood_great
              - mood_unhappy
              - bot_challenge
            responses:
              utter_greet:
              - text: "Hey! How are you?"
              utter_cheer_up:
              - text: "Here is something to cheer you up:"
                image: "https://i.imgur.com/nGF1K8f.jpg"
              utter_did_that_help:
              - text: "Did that help you?"
              utter_happy:
              - text: "Great, carry on!"
              utter_goodbye:
              - text: "Bye"
              utter_iamabot:
              - text: "I am a bot, powered by Rasa."
            session_config:
              session_expiration_time: 60
              carry_over_slots_to_new_session: true
            """
        )
    )
    file = tmp_path / f"{file_name}.yml"
    file.write_text(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            {file_name}:
            - {data_type}: test
              steps:
              - intent: nlu_fallback
              - action: action_two_stage_fallback
              - active_loop: action_two_stage_fallback
           """
    )
    importer = RasaFileImporter(
        config_file=str(config), domain_path=str(domain), training_data_paths=str(file)
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_forms_in_stories_rules()


def test_verify_from_trigger_intent_slot_mapping_not_in_forms_does_not_warn(
    tmp_path: Path,
):
    domain = tmp_path / "domain.yml"
    slot_name = "started_booking_form"
    domain.write_text(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - activate_booking
        entities:
        - city
        slots:
          {slot_name}:
            type: bool
            influence_conversation: false
            mappings:
            - type: from_trigger_intent
              intent: activate_booking
              value: true
          location:
            type: text
            mappings:
            - type: from_entity
              entity: city
        forms:
          booking_form:
            required_slots:
            - location
            """
    )
    importer = RasaFileImporter(domain_path=domain)
    validator = Validator.from_importer(importer)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert validator.verify_slot_mappings()


@pytest.mark.parametrize(
    "config_file, event, message",
    [
        (
            "data/test_config/config_defaults.yml",
            "validator.config_missing_unique_mandatory_key_value",
            "The config file is missing a unique value for "
            "the 'assistant_id' mandatory key.",
        ),
        (
            "data/test_config/config_no_assistant_id.yml",
            "validator.config_missing_mandatory_key",
            "The config file is missing the 'assistant_id' mandatory key.",
        ),
    ],
)
def test_warn_if_config_mandatory_keys_are_not_set_invalid_paths(
    config_file: Text, event: Text, message: Text, capsys: CaptureFixture
) -> None:
    importer = RasaFileImporter(config_file=config_file)
    validator = Validator.from_importer(importer)

    validator.warn_if_config_mandatory_keys_are_not_set()

    result = capsys.readouterr()
    assert message in result.out
    assert event in result.out
    assert "warning" in result.out


@pytest.mark.parametrize(
    "domain_actions, domain_slots, event, log_message",
    [
        # set_slot slot is not listed in the domain
        (
            ["action_transfer_money"],
            {"transfer_amount": {"type": "float", "mappings": []}},
            "validator.verify_flows_steps_against_domain.slot_not_in_domain",
            "The slot 'account_type' is used in the step 'set_account_type' "
            "of flow id 'transfer_money', but it is not listed in the domain slots.",
        ),
        # collect slot is not listed in the domain
        (
            ["action_transfer_money"],
            {"account_type": {"type": "text", "mappings": []}},
            "validator.verify_flows_steps_against_domain.slot_not_in_domain",
            "The slot 'transfer_amount' is used in the step 'ask_amount' "
            "of flow id 'transfer_money', but it is not listed in the domain slots.",
        ),
        # action name is not listed in the domain
        (
            [],
            {
                "account_type": {"type": "text", "mappings": []},
                "transfer_amount": {"type": "float", "mappings": []},
            },
            "validator.verify_flows_steps_against_domain.action_not_in_domain",
            "The action 'action_transfer_money' is used in the step 'execute_transfer' "
            "of flow id 'transfer_money', but it is not listed in the domain file.",
        ),
    ],
)
def test_verify_flow_steps_against_domain_fail(
    tmp_path: Path,
    nlu_data_path: Path,
    domain_actions: List[Text],
    domain_slots: Dict[Text, Any],
    event: Text,
    log_message: Text,
    capsys: CaptureFixture,
) -> None:
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                    flows:
                      transfer_money:
                        description: This flow lets users send money.
                        name: transfer money
                        steps:
                        - id: "ask_amount"
                          collect: transfer_amount
                          next: "set_account_type"
                        - id: "set_account_type"
                          set_slots:
                            - account_type: "debit"
                          next: "execute_transfer"
                        - id: "execute_transfer"
                          action: action_transfer_money
                    """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                    intents:
                      - greet
                    slots:
                        {domain_slots}
                    actions: {domain_actions}
                    """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)
    assert not validator.verify_flows_steps_against_domain()

    result = capsys.readouterr()
    assert log_message in result.out
    assert event in result.out
    assert "error" in result.out


def test_verify_flow_steps_against_domain_disallowed_list_slot(
    tmp_path: Path,
    nlu_data_path: Path,
    capsys: CaptureFixture,
) -> None:
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                flows:
                  order_pizza:
                    description: This flow lets users order their favourite pizza.
                    name: order pizza
                    steps:
                    - id: "ask_pizza_toppings"
                      collect: pizza_toppings
                      next: "ask_address"
                    - id: "ask_address"
                      collect: address
                """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                intents:
                  - greet
                slots:
                    pizza_toppings:
                        type: list
                        mappings: []
                    address:
                        type: text
                        mappings: []
                """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    expected_event = (
        "validator.verify_flows_steps_against_domain" ".use_of_list_slot_in_flow"
    )
    expected_log_level = "error"
    expected_log_message = (
        "The slot 'pizza_toppings' is used in "
        "the step 'ask_pizza_toppings' of flow id "
        "'order_pizza', but it is a list slot. "
        "List slots are currently not supported "
        "in flows."
    )

    assert not validator.verify_flows_steps_against_domain()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_flow_steps_against_domain_interpolated_action_name(
    tmp_path: Path,
    nlu_data_path: Path,
    capsys: CaptureFixture,
) -> None:
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                    flows:
                      pattern_collect_information:
                        description: Test that interpolated names log a warning.
                        name: test flow
                        steps:
                        - id: "validate"
                          action: "validate_{{context.collect}}"
                    """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                    intents:
                      - greet
                    """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    expected_event = (
        "validator.verify_flows_steps_against_domain" ".interpolated_action"
    )
    expected_log_level = "debug"
    expected_log_message = (
        "An interpolated action name 'validate_{context.collect}' "
        "was found at step 'validate' of flow id "
        "'pattern_collect_information'. Skipping validation for "
        "this step."
    )

    assert validator.verify_flows_steps_against_domain()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_unique_flows_duplicate_names(
    tmp_path: Path,
    nlu_data_path: Path,
    capsys: CaptureFixture,
) -> None:
    duplicate_flow_name = "transfer money"
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        flows:
                          transfer_money:
                            description: This flow lets users send money.
                            name: {duplicate_flow_name}
                            steps:
                            - id: "ask_recipient"
                              collect: transfer_recipient
                              next: "ask_amount"
                            - id: "ask_amount"
                              collect: transfer_amount
                              next: "execute_transfer"
                            - id: "execute_transfer"
                              action: action_transfer_money
                          recurrent_payment:
                            description: This flow sets up a recurrent payment.
                            name: {duplicate_flow_name}
                            steps:
                            - id: "set_up_recurrence"
                              action: action_set_up_recurrent_payment
                        """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        intents:
                          - greet
                        slots:
                            transfer_recipient:
                                type: text
                                mappings: []
                            transfer_amount:
                                type: float
                                mappings: []
                        actions:
                          - action_transfer_money
                          - action_set_up_recurrent_payment
                        """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_unique_flows.duplicate_name"
    expected_log_level = "error"
    expected_log_message = (
        f"Detected duplicate flow name '{duplicate_flow_name}' for "
        f"flow id 'recurrent_payment'. Flow names must be unique. "
        f"Please make sure that all flows have different names."
    )

    assert not validator.verify_unique_flows()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_unique_flows_duplicate_descriptions(
    tmp_path: Path,
    nlu_data_path: Path,
    capsys: CaptureFixture,
) -> None:
    duplicate_flow_description_with_punctuation = "This flow lets users send money."
    duplicate_flow_description = "This flow lets users send money"
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        flows:
                          transfer_money:
                            description: {duplicate_flow_description_with_punctuation}
                            name: transfer money
                            steps:
                            - id: "ask_recipient"
                              collect: transfer_recipient
                              next: "ask_amount"
                            - id: "ask_amount"
                              collect: transfer_amount
                              next: "execute_transfer"
                            - id: "execute_transfer"
                              action: action_transfer_money
                          recurrent_payment:
                            description: {duplicate_flow_description}
                            name: setup recurrent payment
                            steps:
                            - id: "set_up_recurrence"
                              action: action_set_up_recurrent_payment
                        """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        intents:
                          - greet
                        slots:
                            transfer_recipient:
                                type: text
                                mappings: []
                            transfer_amount:
                                type: float
                                mappings: []
                        actions:
                          - action_transfer_money
                          - action_set_up_recurrent_payment
                        """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_unique_flows.duplicate_description"
    expected_log_level = "error"
    expected_log_message = (
        "Detected duplicate flow description for flow id "
        "'recurrent_payment'. Flow descriptions must be unique. "
        "Please make sure that all flows have different "
        "descriptions."
    )

    assert not validator.verify_unique_flows()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_verify_predicates_invalid_rejection_if(
    tmp_path: Path,
    nlu_data_path: Path,
    capsys: CaptureFixture,
) -> None:
    predicate = 'slots.account_type not in {{"debit", "savings"}}'

    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        flows:
                          transfer_money:
                            description: This flow lets users send money.
                            name: transfer money
                            steps:
                            - id: "ask_account_type"
                              collect: account_type
                              rejections:
                                - if: {predicate}
                                  utter: utter_invalid_account_type
                              next: "ask_recipient"
                            - id: "ask_recipient"
                              collect: transfer_recipient
                              next: "ask_amount"
                            - id: "ask_amount"
                              collect: transfer_amount
                              next: "execute_transfer"
                            - id: "execute_transfer"
                              action: action_transfer_money
                          recurrent_payment:
                            description: This flow setups recurrent payments
                            name: setup recurrent payment
                            steps:
                            - id: "set_up_recurrence"
                              action: action_set_up_recurrent_payment
                        """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        intents:
                          - greet
                        slots:
                            account_type:
                                type: text
                            transfer_recipient:
                                type: text
                            transfer_amount:
                                type: float
                        actions:
                          - action_transfer_money
                          - action_set_up_recurrent_payment
                        """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_predicates.invalid_rejection"
    expected_log_level = "error"
    expected_log_message = (
        f"Detected invalid rejection '{predicate}' "
        f"at `collect` step 'ask_account_type' for "
        f"flow id 'transfer_money'. Please make sure "
        f"that all conditions are valid."
    )

    assert not validator.verify_predicates()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


def test_flow_predicate_validation_fails_for_faulty_flow_link_predicates():
    flows = flows_from_str(
        """
        flows:
          pattern_bar:
            description: Test that faulty flow link predicates are detected.
            steps:
            - id: first
              action: action_listen
              next:
                - if: xxx !!!
                  then: END
                - else: END
        """
    )
    validator = Validator(Domain.empty(), TrainingData(), StoryGraph([]), flows, None)
    assert not validator.verify_predicates()


def test_verify_predicates_with_valid_jinja(
    tmp_path: Path,
    nlu_data_path: Path,
) -> None:
    predicate_collect = '"{{context.collect}} is not null"'
    predicate_link = '"{{context.collect}} is null"'
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        flows:
                          transfer_money:
                            description: This flow lets users send money.
                            name: transfer money
                            steps:
                            - id: "ask_account_type"
                              collect: account_type
                              rejections:
                                - if: {predicate_collect}
                                  utter: utter_invalid_account_type
                              next: "ask_recipient"
                            - id: "ask_recipient"
                              collect: transfer_recipient
                              next:
                                - if: {predicate_link}
                                  then: "ask_amount"
                                - else: END
                            - id: "ask_amount"
                              collect: transfer_amount
                              next: "execute_transfer"
                            - id: "execute_transfer"
                              action: action_transfer_money
                        """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                        intents:
                          - greet
                        slots:
                            transfer_recipient:
                                type: text
                                mappings: []
                            transfer_amount:
                                type: float
                                mappings: []
                        actions:
                          - action_transfer_money
                        """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    assert validator.verify_predicates()


@pytest.fixture
def domain_file_name(tmp_path: Path) -> Path:
    domain_file_name = tmp_path / "domain.yml"
    with open(domain_file_name, "w") as file:
        file.write(
            f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                responses:
                  utter_ask_recipient:
                    - text: "Who do you want to send money to?"
                  utter_ask_amount:
                    - text: "How much do you want to send?"
                  utter_amount_too_high:
                    - text: "Sorry, you can only send up to 1000."
                  utter_transfer_summary:
                    - text: You are sending {{amount}} to {{transfer_recipient}}.
                """
        )
    return domain_file_name


@pytest.mark.parametrize("predicate", ["account_type is null", "not account_type"])
def test_verify_predicates_namespaces_not_referenced(
    predicate: str,
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_bar:
            description: Test that predicates without namespaces are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "{predicate}"
                  then: END
                - else: END
        """
    )

    expected_event = (
        "validator.verify_namespaces" ".referencing_variables_without_namespace"
    )
    expected_log_level = "error"
    expected_log_message = (
        f"Predicate '{predicate}' at step 'first' for flow id "
        f"'flow_bar' references one or more variables  without "
        f"the `slots.` or `context.` namespace prefix. "
        f"Please make sure that all variables reference the required "
        f"namespace."
    )
    validator = Validator(Domain.empty(), TrainingData(), StoryGraph([]), flows, None)
    assert not validator.verify_predicates()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


@pytest.mark.parametrize(
    "predicate, expected_validation_result",
    [
        ("True", True),
        ("False", True),
        ("slots.spam", True),
        ("slots.spam is 'eggs'", True),
        ("slots.authenticated AND slots.email_verified", True),
        ("slots.authenticated OR slots.email_verified", True),
        ("xxx !!!", False),
    ],
)
def test_verify_predicates_on_flow_guards(
    predicate: str, expected_validation_result: bool
):
    """Test that verify_predicates() correctly verify flow guard predicates."""
    # Given
    flows = flows_from_str(
        f"""
        flows:
          spam_eggs:
            description: Test that predicates are validated.
            if: {predicate}
            steps:
            - id: first
              action: action_listen
        """
    )
    validator = Validator(Domain.empty(), TrainingData(), StoryGraph([]), flows, None)
    # When
    validation_result = validator.verify_predicates()
    # Then
    assert validation_result == expected_validation_result


@pytest.mark.parametrize(
    "predicate",
    [
        "xxx !!!",
        "slots.spam is 'eggs' AND",
        "slots.spam is 'eggs' OR",
        "slots.spam is AND OR not 'eggs'",
    ],
)
def test_verify_predicates_invalid_flow_guards(
    predicate: str,
    capsys: CaptureFixture,
) -> None:
    """Test that verify_predicates() correctly logs invalid flow guard predicates."""
    # Given
    expected_log_event = "validator.verify_predicates.flow_guard.invalid_condition"
    expected_log_level = "error"
    expected_log_message = (
        f"Detected invalid flow guard condition "
        f"'{predicate}' for flow id 'spam_eggs'. "
        f"Please make sure that all conditions are valid."
    )
    flows = flows_from_str(
        f"""
        flows:
          spam_eggs:
            description: Test that predicates are validated.
            if: {predicate}
            steps:
            - id: first
              action: action_listen
        """
    )
    validator = Validator(Domain.empty(), TrainingData(), StoryGraph([]), flows, None)

    assert not validator.verify_predicates()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_log_event in result.out
    assert expected_log_level in result.out


@pytest.mark.parametrize(
    "predicate",
    [
        "slots.account_type is 'debit'",
        "not slots.account_type",
        "context.collect is not null",
        "not context.collect",
    ],
)
def test_verify_predicates_reference_namespaces(
    predicate: str,
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_bar:
            description: Test that predicates with namespaces are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "{predicate}"
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          account_type:
            type: text
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert validator.verify_predicates()

    result = capsys.readouterr()
    assert "error" not in result.out


@pytest.mark.parametrize(
    "predicate",
    [
        "{'credit' 'debit'} contains slots.account_type",
        "slots.account_type is 'debit'",
        "slots.account_type == 'debit'",
        "slots.account_type != 'debit'",
        "not slots.account_type",
        "context.collect is not null",
        "not context.collect",
    ],
)
def test_verify_categorical_predicate_valid_value(
    predicate: str, capsys: CaptureFixture
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_bar:
            description: Test that values in checks for categorical slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "{predicate}"
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          account_type:
            type: categorical
            values:
              - credit
              - debit
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert validator.verify_predicates()
    result = capsys.readouterr()
    assert "error" not in result.out


@pytest.mark.parametrize(
    "predicate",
    [
        "slots.account_type is savings",
        "slots.account_type == savings",
        "slots.account_type != savings",
        "{'savings' 'investment'} contains slots.account_type",
    ],
)
def test_verify_categorical_predicate_invalid_value(
    predicate: str, capsys: CaptureFixture
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_bar:
            description: Test that values in checks for categorical slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "{predicate}"
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          account_type:
            type: categorical
            values:
              - credit
              - debit
            mappings: []
        """
    )
    expected_log_level = "error"
    expected_log_event = "validator.verify_predicates.link.invalid_condition"
    expected_log_message_parts = [
        f"Detected invalid condition '{predicate}' ",
        "at step 'first' for flow id 'flow_bar'. ",
        "Please make sure that all conditions are valid.",
    ]
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)
    assert not validator.verify_predicates()

    result = capsys.readouterr()
    assert expected_log_message_parts[0] in result.out
    assert expected_log_event in result.out
    assert expected_log_level in result.out


def test_verify_categorical_predicate_with_apostrophe_valid(
    capsys: CaptureFixture,
) -> None:
    """checks that a categorical slot with apostrophe is valid."""
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test that values in checks for categorical slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: slots.account_type == "don't know"
                  then: END
                - else: END
          flow_bar2:
            description: Test that values in checks for categorical slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: slots.account_type == "dont know'"
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          account_type:
            type: categorical
            values:
              - don't know
              - dont know'
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert validator.verify_predicates()

    result = capsys.readouterr()
    assert "error" not in result.out


def test_verify_categorical_predicate_with_double_quotes_valid() -> None:
    """checks that a categorical slot with double quotes is invalid."""
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test that values in checks for categorical slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: slots.account_type == 'don"t know'
                  then: END
                - else: END
          flow_bar2:
            description: Test that values in checks for categorical slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: slots.account_type == 'dont know"'
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          account_type:
            type: categorical
            values:
              - don"t know
              - dont know"
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    with structlog.testing.capture_logs() as caplog:
        assert validator.verify_predicates()
        logs = filter_logs(caplog, log_level="error")
        assert len(logs) == 0


@pytest.mark.parametrize(
    "predicate",
    [
        "slots.confirmation",
        "not slots.confirmation",
        "slots.confirmation == true",
        "slots.confirmation is not true",
        "not slots.confirmation == true",
    ],
)
def test_verify_boolean_predicate_valid_value(predicate: str) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_bar:
            description: Test that values in checks for boolean slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "{predicate}"
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          confirmation:
            type: bool
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    with structlog.testing.capture_logs() as caplog:
        assert validator.verify_predicates()
        logs = filter_logs(caplog, log_level="error")
        assert len(logs) == 0


@pytest.mark.parametrize(
    "predicate",
    [
        "slots.confirmation == test",
        "slots.confirmation is not test",
    ],
)
def test_verify_boolean_predicate_invalid_value(
    predicate: str, capsys: CaptureFixture
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_bar:
            description: Test that values in checks for boolean slots are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "{predicate}"
                  then: END
                - else: END
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          confirmation:
            type: bool
            mappings: []
        """
    )
    expected_log_level = "error"
    expected_log_event = "validator.verify_predicates.link.invalid_condition"
    expected_log_message_parts = [
        f"Detected invalid condition '{predicate}' ",
        "at step 'first' for flow id 'flow_bar'. ",
        "Please make sure that all conditions are valid.",
    ]
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)
    assert not validator.verify_predicates()

    result = capsys.readouterr()
    assert expected_log_message_parts[0] in result.out
    assert expected_log_event in result.out
    assert expected_log_level in result.out


def test_verify_namespaces_reference_slots_not_in_the_domain(
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test that slots referenced in predicates are validated.
            steps:
            - id: first
              action: action_listen
              next:
                - if: "slots.membership is 'gold'"
                  then: END
                - else: END
        """
    )
    expected_log_level = "error"
    expected_log_event = "validator.verify_namespaces.invalid_slot"
    expected_log_message = (
        "Detected invalid slot 'membership' "
        "at step 'first' for flow id 'flow_bar'. "
        "Please make sure that all slots are specified "
        "in the domain file."
    )
    validator = Validator(Domain.empty(), TrainingData(), StoryGraph([]), flows, None)
    assert not validator.verify_predicates()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_log_event in result.out
    assert expected_log_level in result.out


def test_verify_flow_steps_against_domain_disallows_collect_step_with_action_utterance(
    tmp_path: Path,
    nlu_data_path: Path,
    capsys: CaptureFixture,
) -> None:
    flows_file = tmp_path / "flows.yml"
    with open(flows_file, "w") as file:
        file.write(
            f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                flows:
                  order_pizza:
                    description: This flow lets users order their favourite pizza.
                    name: order pizza
                    steps:
                    - id: "ask_pizza_toppings"
                      collect: pizza_toppings
                      next: END
                """
        )
    domain_file = tmp_path / "domain.yml"
    with open(domain_file, "w") as file:
        file.write(
            f"""
                version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
                actions:
                    - action_ask_pizza_toppings
                responses:
                  utter_ask_pizza_toppings:
                    - text: "What toppings do you want?"
                """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=str(domain_file),
        training_data_paths=[str(flows_file), str(nlu_data_path)],
    )

    validator = Validator.from_importer(importer)

    expected_event = "validator.verify_flows_steps_against_domain.collect_step"
    expected_log_level = "error"
    expected_log_message = (
        "The collect step 'pizza_toppings' has an utterance "
        "'utter_ask_pizza_toppings' as well as an action "
        "'action_ask_pizza_toppings' defined. "
        "You can just have one of them! "
        "Please remove either the utterance or the action."
    )

    assert not validator.verify_flows_steps_against_domain()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_event in result.out
    assert expected_log_level in result.out


@pytest.mark.parametrize(
    "response, result",
    [
        ("Hello {name}", False),
        ("Hello {}", True),
        ("Hello { }", True),
        ("Hello {    }", True),
        ("Hello { name }", False),
        (["Hello {name}", "{}"], True),
        ({"key": "Hello {name}"}, False),
        ({"key": "{}"}, True),
        ({"key": ["Hello {name}", "{}"]}, True),
        ({"key": {"key": "Hello {name}"}}, False),
        ({"key": {"key": "{}"}}, True),
        ({"key": {"key": ["Hello {name}", "{}"]}}, True),
        (
            [{"key": {"key": ["Hello {name}", "{}"], "key2": ["Hello {name}", "{}"]}}],
            True,
        ),
    ],
)
def test_validator_check_for_placeholder(
    response: Union[str, List, Dict], result: bool
) -> None:
    assert Validator.check_for_placeholder(response) is result


def test_validator_check_for_empty_parenthesis_in_text_response() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_greet:
            - text: "Hey! How are you? {{name}}{{}}"
            utter_did_that_help:
            - text: "Did that help you?"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False


def test_validator_check_for_empty_parenthesis_in_image_response() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_cheer_up:
            - text: "Here is something to cheer you up:"
              image: "https://i.imgur.com/nGF1K8f.jpg{{}}"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False


def test_validator_check_for_empty_parenthesis_in_button_response() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_ask_confirm:
            - buttons:
                - payload: "yes"
                  title: Yes
                - payload: "{{}}"
                  title: No
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False


def test_validator_check_for_empty_parenthesis_in_text_button_response() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_ask_confirm:
            - buttons:
                - payload: "yes"
                  title: Yes
                - payload: "no"
                  title: "{{}}"
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False


def test_validator_check_for_empty_parenthesis_in_custom_response() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_ask_custom:
            - custom:
                field: "slot_value"
                properties:
                    field_prefixed: "test {{}}"
                bool_field: true
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False


def test_validator_check_for_empty_parenthesis_multiple_errors() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_greet:
            - text: "Hey! How are you? {{}}"
            utter_did_that_help:
            - text: "Did that help you?"
            utter_cheer_up:
            - text: "Here is something to cheer you up:"
              image: "https://i.imgur.com/nGF1K8f.jpg{{}}"
            utter_ask_confirm:
            - buttons:
                - payload: "yes"
                  title: Yes
                - payload: "{{}}"
                  title: No
              text: "Do you confirm? {{}}"
            utter_ask_custom:
            - custom:
                field: "slot_value"
                properties:
                    field_prefixed: "test {{}}"
                bool_field: true
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False


def test_validator_check_for_empty_parenthesis_all_good() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_greet:
            - text: "Hey! How are you?"
            utter_did_that_help:
            - text: "Did that help you?"
            utter_cheer_up:
            - text: "Here is something to cheer you up:"
              image: "https://i.imgur.com/nGF1K8f.jpg"
            utter_ask_confirm:
            - buttons:
                - payload: "yes"
                  title: Yes
                - payload: "no"
                  title: No
              text: "Do you confirm?"
            utter_ask_custom:
            - custom:
                field: "slot_value"
                properties:
                    field_prefixed: "test"
                bool_field: true
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is True


def test_validator_check_for_empty_parenthesis_empty_response(
    capsys: CaptureFixture,
) -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_greet: []
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.check_for_no_empty_parenthesis_in_responses() is False

    captured = capsys.readouterr()
    assert (
        "The response 'utter_greet' in the domain file "
        "does not have any variations. Please add at least one "
        "variation to the response." in captured.out
    )


def test_validator_fail_as_both_utterance_and_action_defined_for_collect(
    capsys: CaptureFixture,
) -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        actions:
            - action_ask_transfer_amount
        responses:
            utter_ask_transfer_amount:
                - text: "How much money should I send?"
        slots:
            transfer_amount:
                type: float
                mappings: []
        """
    )
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test flow.
            steps:
            - collect: transfer_amount
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    expected_log_level = "error"
    expected_log_event = "validator.verify_flows_steps_against_domain.collect_step"
    expected_log_message = (
        "The collect step 'transfer_amount' has an utterance "
        "'utter_ask_transfer_amount' as well as an action "
        "'action_ask_transfer_amount' defined. "
        "You can just have one of them! "
        "Please remove either the utterance or the action."
    )
    assert validator.verify_flows_steps_against_domain() is False

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_log_event in result.out
    assert expected_log_level in result.out


def test_validator_fail_as_both_utterance_and_action_not_defined_for_collect(
    capsys: CaptureFixture,
) -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
            transfer_amount:
                type: float
                mappings: []
        """
    )
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test flow.
            steps:
            - collect: transfer_amount
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    expected_log_level = "error"
    expected_log_event = "validator.verify_flows_steps_against_domain.collect_step"
    expected_log_message = (
        "The collect step 'transfer_amount' has neither an utterance "
        "nor an action defined, or an initial value defined in the domain."
        "You need to define either an utterance or an action."
    )
    assert validator.verify_flows_steps_against_domain() is False

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_log_event in result.out
    assert expected_log_level in result.out


def test_validator_pass_as_only_utterance_defined_for_collect() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_ask_transfer_amount:
                - text: "How much money should I send?"
        slots:
            transfer_amount:
                type: float
                mappings: []
        """
    )
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test flow.
            steps:
            - collect: transfer_amount
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)
    assert validator.verify_flows_steps_against_domain() is True


def test_validator_pass_as_only_action_defined_for_collect() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        actions:
            - action_ask_transfer_amount
        slots:
            transfer_amount:
                type: float
                mappings: []
        """
    )
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test flow.
            steps:
            - collect: transfer_amount
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)
    assert validator.verify_flows_steps_against_domain() is True


def test_validator_pass_as_initial_slot_value_defined_for_collect() -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
            transfer_amount:
                type: float
                initial_value: 100
                mappings: []
        """
    )
    flows = flows_from_str(
        """
        flows:
          flow_bar:
            description: Test flow.
            steps:
            - collect: transfer_amount
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)
    assert validator.verify_flows_steps_against_domain() is True


def test_validate_button_payloads_no_payload(capsys: CaptureFixture) -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
            utter_ask_confirm:
            - buttons:
                - title: Yes
                - title: No
                  payload: " "
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.validate_button_payloads() is False

    captured = capsys.readouterr()
    assert (
        "The button 'Yes' in response 'utter_ask_confirm' does "
        "not have a payload." in captured.out
    )
    assert (
        "The button 'No' in response 'utter_ask_confirm' has "
        "an empty payload." in captured.out
    )


def test_validate_button_payloads_free_form_payloads(capsys: CaptureFixture) -> None:
    test_domain = Domain.from_yaml(
        """
        responses:
            utter_ask_confirm:
            - buttons:
                - title: Yes
                  payload: yes
                - title: No
                  payload: no
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.validate_button_payloads() is True

    captured = capsys.readouterr()
    logging_level = "warning"
    logging_message = (
        "Using a free form string in payload of a button "
        "implies that the string will be sent to the NLU "
        "interpreter for parsing. To avoid the need for "
        "parsing at runtime, it is recommended to use "
        "one of the documented formats "
        "(https://rasa.com/docs/rasa-pro/concepts/responses#buttons)"
    )
    assert logging_level in captured.out
    assert logging_message in captured.out


@pytest.mark.parametrize(
    "payload", ["/SetSlots(confirmation=True)", '/inform{{"confirmation": "true"}}']
)
def test_validate_button_payloads_valid_payloads(
    capsys: CaptureFixture, payload: str
) -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
         - inform
        entities:
        - confirmation
        slots:
          confirmation:
             type: bool
             mappings:
              - type: from_entity
                entity: confirmation
        responses:
            utter_ask_confirm:
            - buttons:
                - title: Yes
                  payload: '{payload}'
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.validate_button_payloads() is True

    captured = capsys.readouterr()
    log_levels = ["error", "warning"]
    assert all([log_level not in captured.out for log_level in log_levels])


def test_validate_button_payloads_no_user_warning_raised_with_intent_payload() -> None:
    """Test that no user warning is raised when the payload has double curly braces."""
    payload = '/inform{{"confirmation": "true"}}'
    test_domain = Domain.from_yaml(
        f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
             - inform
            entities:
            - confirmation
            slots:
              confirmation:
                 type: bool
                 mappings:
                  - type: from_entity
                    entity: confirmation
            responses:
                utter_ask_confirm:
                - buttons:
                    - title: Yes
                      payload: '{payload}'
                  text: "Do you confirm?"
            """
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
        assert validator.validate_button_payloads() is True


@pytest.mark.parametrize(
    "payload",
    ['/inform[["confirmation": "false"]]', '/SetSlots["confirmation": "false"]'],
)
def test_validate_button_payloads_invalid_payloads(
    capsys: CaptureFixture, payload: str
) -> None:
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - inform
        responses:
            utter_ask_confirm:
            - buttons:
                - title: No
                  payload: '{payload}'
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.validate_button_payloads() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert (
        "The button 'No' in response "
        "'utter_ask_confirm' does not follow valid payload formats "
        "for triggering a specific intent and entities or for "
        "triggering a SetSlot command."
    ) in captured.out


def test_validate_button_payloads_above_slot_limit(capsys: CaptureFixture) -> None:
    payload = "/SetSlots(" + "test_slot=1, " * 11 + ")"
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - inform
        slots:
          test_slot:
             type: float
        responses:
            utter_ask_confirm:
            - buttons:
                - title: Test
                  payload: '{payload}'
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.validate_button_payloads() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert "validator.validate_button_payloads.slot_limit_exceeded" in captured.out
    assert (
        "The button 'Test' in response 'utter_ask_confirm' has a payload "
        "that sets more than 10 slots. Please make sure that the number "
        "of slots set by the button payload does not exceed the limit."
    ) in captured.out


def test_validate_button_payloads_unique_slot_names(capsys: CaptureFixture) -> None:
    payload = "/SetSlots(name=John, name=Paul)"
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        intents:
        - inform
        slots:
          name:
             type: text
        responses:
            utter_ask_name:
            - buttons:
                - title: Name
                  payload: '{payload}'
              text: "Do you confirm?"
        """
    )

    validator = Validator(test_domain, TrainingData(), StoryGraph([]), None, None)
    assert validator.validate_button_payloads() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert "validator.validate_button_payloads.duplicate_slot_name" in captured.out
    assert (
        "The button 'Name' in response 'utter_ask_name' has a command "
        "to set the slot 'name' multiple times. Please make sure "
        "that each slot is set only once."
    ) in captured.out


def test_validate_CALM_slot_mappings_success(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        config_file="data/test_calm_slot_mappings/config.yml",
        domain_path="data/test_calm_slot_mappings/validation/domain_valid.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/data/flows.yml",
            "data/test_calm_slot_mappings/data/nlu.yml",
            "data/test_calm_slot_mappings/data/stories.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is True

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level not in captured.out


@pytest.mark.parametrize(
    "domain_path",
    [
        "data/test_calm_slot_mappings/validation/domain_with_llm_and_custom_mappings.yml",
        "data/test_calm_slot_mappings/validation/domain_with_llm_and_nlu_mappings.yml",
    ],
)
def test_domain_slots_contain_all_mapping_type(
    capsys: CaptureFixture, domain_path: str
) -> None:
    importer = RasaFileImporter(
        config_file="data/test_calm_slot_mappings/config.yml",
        domain_path=domain_path,
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is True


def test_validate_custom_action_defined_in_the_domain(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        config_file="data/test_calm_slot_mappings/config.yml",
        domain_path="data/test_calm_slot_mappings/validation/domain_custom_action_missing.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert (
        "validator.validate_slot_mappings_in_CALM.custom_action_not_in_domain"
        in captured.out
    )

    assert (
        "The slot 'card_number' has a custom action 'action_set_card_number' "
        "defined in its slot mappings, but the "
        "action is not listed in the domain actions. "
        "Please add the action to your domain file."
    ) in captured.out


def test_validate_nlu_command_adapter_not_in_config(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        config_file="data/test_calm_slot_mappings/validation/config_nlu_command_adapter_missing.yml",
        domain_path="data/test_calm_slot_mappings/validation/domain_valid_nlu_mappings.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert (
        "validator.validate_slot_mappings_in_CALM.nlu_mappings_without_adapter"
        in captured.out
    )
    assert (
        "The slot 'card_number' has NLU slot mappings, "
        "but the NLUCommandAdapter is not present in the "
        "pipeline. Please add the NLUCommandAdapter to the "
        "pipeline in the config file."
    ) in captured.out


def test_validate_llm_slot_mappings_in_nlu_based_assistant(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        domain_path="data/test_calm_slot_mappings/validation/domain_valid_llm_mappings.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/stories.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert (
        "validator.validate_slot_mappings_in_CALM.llm_mappings_without_flows"
        in captured.out
    )
    assert (
        "The slot 'num_people' has LLM slot mappings, "
        "but no flows are present in the training data files. "
        "Please add flows to the training data files."
    ) in captured.out


def test_validate_llm_slot_mapping_with_action_ask_success(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        domain_path="data/test_calm_slot_mappings/validation/domain_valid_llm_mappings.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is True

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level not in captured.out


def test_validate_custom_slot_mappings_with_action_property_success(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        domain_path="data/test_calm_slot_mappings/validation/domain_custom_mappings_valid.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows_for_valid_custom_slot_mappings.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is True

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level not in captured.out


def test_verify_slot_persistence_configuration_duplicate(
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that duplicate slot persistence configs are validated.
            persisted_slots:
            - slot_a
            steps:
            - collect: slot_a
              reset_after_flow_ends: false
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    expected_log_level = "error"
    expected_log_event = (
        "validator.verify_slot_persistence_configuration.duplicate_config"
    )
    expected_log_message = (
        "Flow with id 'flow_a' uses the 'reset_after_flow_ends' property "
        "in collect step 'slot_a' and also the "
        "'persisted_slots' property at the flow level. "
        "Please use only one of the two configuration methods."
    )

    assert not validator.verify_slot_persistence_configuration()

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_log_level in result.out
    assert expected_log_event in result.out


@patch("rasa.telemetry._track")
def test_verify_slot_persistence_configuration_invalid_slots(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that duplicate slot persistence configs are validated.
            persisted_slots:
            - slot_a
            - invalid_slot
            steps:
            - collect: slot_a
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    expected_log_level = "error"
    expected_log_event = (
        "validator.verify_slot_persistence_configuration.invalid_persist_slot"
    )
    expected_log_message = (
        "Flow with id 'flow_a' lists slot(s) '{'invalid_slot'}' in the "
        "'persisted_slots' property. However these slots are "
        "neither used in a collect step nor a set_slot step of the flow. "
        "Please remove such slots from the 'persisted_slots' property."
    )

    assert not validator.verify_slot_persistence_configuration()

    mock_track.assert_called_once_with(
        TELEMETRY_VALIDATION_ERROR_LOG_EVENT,
        {
            "flow": "flow_a",
            "message": expected_log_message,
            "log_id": expected_log_event,
            "log_level": expected_log_level,
        },
    )

    result = capsys.readouterr()
    assert expected_log_message in result.out
    assert expected_log_level in result.out
    assert expected_log_event in result.out


def test_verify_slot_persistence_configuration_raises_deprecation_warning() -> None:
    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that duplicate slot persistence configs are validated.
            steps:
            - collect: slot_a
              reset_after_flow_ends: false
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    deprecation_message = (
        "Configuring 'reset_after_flow_ends' in collect step 'slot_a' is deprecated "
        "and will be removed in Rasa Pro 4.0.0. In flow id 'flow_a', please use "
        "the 'persisted_slots' property at the flow level instead."
    )

    with pytest.warns(FutureWarning) as record:
        assert validator.verify_slot_persistence_configuration()

    assert len(record) == 1
    assert record[0].message.args[0] == deprecation_message
    assert isinstance(record[0].message, FutureWarning)


def test_validate_allow_nlu_correction_valid() -> None:
    importer = RasaFileImporter(
        config_file="data/test_calm_slot_mappings/config.yml",
        domain_path="data/test_calm_slot_mappings/validation/domain_with_valid_allow_nlu_correction.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is True


def test_validate_allow_nlu_correction_invalid(
    capsys: CaptureFixture,
) -> None:
    importer = RasaFileImporter(
        config_file="data/test_calm_slot_mappings/config.yml",
        domain_path="data/test_calm_slot_mappings/validation/domain_with_invalid_nlu_correction.yml",
        training_data_paths=[
            "data/test_calm_slot_mappings/validation/flows.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.validate_CALM_slot_mappings() is False

    captured = capsys.readouterr()
    log_level = "error"
    assert log_level in captured.out
    assert (
        "The slot 'card_number' has at least 1 slot mapping with "
        "'allow_nlu_correction' set to 'true', "
        "but the slot mapping type is not 'from_llm'. "
        "Please set the slot mapping type to 'from_llm' "
        "to allow the LLM to correct this slot."
    ) in captured.out
    assert (
        "The slot 'num_people' does not have any NLU-based slot mappings. "
        "The property `allow_nlu_correction` is only applicable when the "
        "slot contains both NLU-based and LLM-based slot mappings."
    ) in captured.out


@pytest.mark.parametrize(
    "ask_confirm_digressions, block_digressions",
    [
        (True, True),
        ("[flow_b]", "[flow_c]"),
    ],
)
def test_verify_digression_configuration_at_step_level_valid(
    ask_confirm_digressions: Any, block_digressions: Any
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_a:
            description: Test that digressions properties are valid.
            steps:
            - collect: slot_a
              ask_confirm_digressions: {ask_confirm_digressions}
              block_digressions: {block_digressions}
          flow_b:
            description: Test flow B.
            steps:
            - action: utter_welcome
          flow_c:
            description: Test flow C.
            steps:
            - action: utter_chitchat
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        responses:
            utter_welcome:
                - text: "Welcome!"
            utter_chitchat:
                - text: "Chitchat!"
            utter_ask_slot_a:
                - text: "What is slot A?"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert validator.verify_digression_configuration() is True


@pytest.mark.parametrize(
    "ask_confirm_digressions, block_digressions",
    [
        (True, True),
        ("[flow_b]", "[flow_c]"),
    ],
)
def test_verify_digression_configuration_at_flow_level_valid(
    ask_confirm_digressions: Any, block_digressions: Any
) -> None:
    flows = flows_from_str(
        f"""
        flows:
          flow_a:
            description: Test that digressions properties are valid.
            ask_confirm_digressions: {ask_confirm_digressions}
            block_digressions: {block_digressions}
            steps:
            - collect: slot_a
          flow_b:
            description: Test flow B.
            steps:
            - action: utter_welcome
          flow_c:
            description: Test flow C.
            steps:
            - action: utter_chitchat
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        responses:
            utter_welcome:
                - text: "Welcome!"
            utter_chitchat:
                - text: "Chitchat!"
            utter_ask_slot_a:
                - text: "What is slot A?"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert validator.verify_digression_configuration() is True


def test_verify_digression_configuration_at_step_level_invalid(
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that digressions properties are valid.
            steps:
            - collect: slot_a
              id: collect_slot_a
              ask_confirm_digressions:
                - flow_b
              block_digressions:
                - flow_c
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        responses:
            utter_ask_slot_a:
                - text: "What is slot A?"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert not validator.verify_digression_configuration()

    expected_log_messages = [
        "The flow 'flow_b' is listed in the `ask_confirm_digressions` "
        "configuration of step 'collect_slot_a' in flow 'flow_a', but it is "
        "not found in the flows file. Please make sure that the flow id is correct.",
        "The flow 'flow_c' is listed in the `block_digressions` "
        "configuration of step 'collect_slot_a' in flow 'flow_a', but it is "
        "not found in the flows file. Please make sure that the flow id is correct.",
    ]

    captured = capsys.readouterr()

    assert any(
        [
            expected_log_message in captured.out
            for expected_log_message in expected_log_messages
        ]
    )
    assert "validator.verify_digression_configuration" in captured.out
    assert "error" in captured.out


def test_verify_digression_configuration_at_flow_level_invalid(
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that digressions properties are valid.
            ask_confirm_digressions:
             - flow_b
            block_digressions:
             - flow_c
            steps:
            - collect: slot_a
              id: collect_slot_a
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        responses:
            utter_ask_slot_a:
                - text: "What is slot A?"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert not validator.verify_digression_configuration()

    expected_log_messages = [
        "The flow 'flow_b' is listed in the `ask_confirm_digressions` "
        "configuration of flow 'flow_a', but it is "
        "not found in the flows file. Please make sure that the flow id is correct.",
        "The flow 'flow_c' is listed in the `block_digressions` "
        "configuration of flow 'flow_a', but it is "
        "not found in the flows file. Please make sure that the flow id is correct.",
    ]

    captured = capsys.readouterr()

    assert any(
        [
            expected_log_message in captured.out
            for expected_log_message in expected_log_messages
        ]
    )
    assert "validator.verify_digression_configuration" in captured.out
    assert "error" in captured.out


def test_verify_digression_configuration_at_step_level_invalid_duplicate(
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that digressions properties are valid.
            steps:
            - collect: slot_a
              id: collect_slot_a
              ask_confirm_digressions:
                - flow_b
              block_digressions:
                - flow_b
          flow_b:
            description: Test flow B.
            steps:
            - action: utter_welcome
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        responses:
            utter_ask_slot_a:
                - text: "What is slot A?"
            utter_welcome:
                - text: "Welcome!"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert not validator.verify_digression_configuration()

    expected_log_message = (
        "The flow 'flow_b' is listed in both the "
        "`ask_confirm_digressions` and `block_digressions` "
        "configuration of step 'collect_slot_a' in flow 'flow_a'. "
        "Please make sure that the flow id is not listed in both "
        "configurations."
    )

    captured = capsys.readouterr()

    assert expected_log_message in captured.out
    assert "validator.verify_digression_configuration" in captured.out
    assert "error" in captured.out


def test_verify_digression_configuration_at_flow_level_invalid_duplicate(
    capsys: CaptureFixture,
) -> None:
    flows = flows_from_str(
        """
        flows:
          flow_a:
            description: Test that digressions properties are valid.
            ask_confirm_digressions:
             - flow_b
            block_digressions:
             - flow_b
            steps:
            - collect: slot_a
              id: collect_slot_a
          flow_b:
            description: Test flow B.
            steps:
            - action: utter_welcome
        """
    )
    test_domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          slot_a:
            type: text
            mappings: []
        responses:
            utter_ask_slot_a:
                - text: "What is slot A?"
            utter_welcome:
                - text: "Welcome!"
        """
    )
    validator = Validator(test_domain, TrainingData(), StoryGraph([]), flows, None)

    assert not validator.verify_digression_configuration()

    expected_log_message = (
        "The flow 'flow_b' is listed in both the "
        "`ask_confirm_digressions` and `block_digressions` "
        "configuration of flow 'flow_a'. "
        "Please make sure that the flow id is not listed in both "
        "configurations."
    )

    captured = capsys.readouterr()

    assert expected_log_message in captured.out
    assert "validator.verify_digression_configuration" in captured.out
    assert "error" in captured.out
