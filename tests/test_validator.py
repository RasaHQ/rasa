from typing import Text

import pytest
from _pytest.logging import LogCaptureFixture

from rasa.validator import Validator
from rasa.shared.importers.rasa import RasaFileImporter
from rasa.shared.importers.autoconfig import TrainingType
from pathlib import Path


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


def test_verify_nlu_with_e2e_story(tmp_path: Path, nlu_data_path: Path):
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
            - story: path 2
              steps:
              - intent: greet
              - action: utter_greet
            """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path="data/test_moodbot/domain.yml",
        training_data_paths=[story_file_name, nlu_data_path],
        training_type=TrainingType.NLU,
    )

    validator = Validator.from_importer(importer)
    assert validator.verify_nlu()


def test_verify_intents_does_not_fail_on_valid_data(nlu_data_path: Text):
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path],
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_intents()


def test_verify_intents_does_fail_on_invalid_data(nlu_data_path: Text):
    # domain and nlu data are from different domain and should produce warnings
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=[nlu_data_path],
    )
    validator = Validator.from_importer(importer)
    assert not validator.verify_intents()


def test_verify_valid_responses():
    importer = RasaFileImporter(
        domain_path="data/test_domains/selectors.yml",
        training_data_paths=[
            "data/test_selectors/nlu.yml",
            "data/test_selectors/stories.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_utterances_in_stories()


def test_verify_valid_responses_in_rules(nlu_data_path: Text):
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=[
            nlu_data_path,
            "data/test_yaml_stories/rules_without_stories_and_wrong_names.yml",
        ],
    )
    validator = Validator.from_importer(importer)
    assert not validator.verify_utterances_in_stories()


def test_verify_story_structure(stories_path: Text):
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml", training_data_paths=[stories_path],
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_story_structure(ignore_warnings=False)


def test_verify_bad_story_structure():
    importer = RasaFileImporter(
        domain_path="data/test_domains/default.yml",
        training_data_paths=["data/test_yaml_stories/stories_conflicting_2.yml"],
    )
    validator = Validator.from_importer(importer)
    assert not validator.verify_story_structure(ignore_warnings=False)


def test_verify_bad_e2e_story_structure_when_text_identical(tmp_path: Path):
    story_file_name = tmp_path / "stories.yml"
    story_file_name.write_text(
        """
        version: "3.0"
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
        training_type=TrainingType.NLU,
    )
    validator = Validator.from_importer(importer)
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
        training_type=TrainingType.NLU,
    )
    validator = Validator.from_importer(importer)
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
        training_type=TrainingType.NLU,
    )
    validator = Validator.from_importer(importer)
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
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path],
    )
    validator = Validator.from_importer(importer)
    assert not validator.verify_example_repetition_in_intents(False)


def test_verify_logging_message_for_intent_not_used_in_nlu(
    caplog: LogCaptureFixture, validator_under_test: Validator
):
    caplog.clear()
    with pytest.warns(UserWarning) as record:
        validator_under_test.verify_intents(False)

    assert (
        "The intent 'goodbye' is listed in the domain file, "
        "but is not found in the NLU training data."
        in (m.message.args[0] for m in record)
    )


def test_verify_logging_message_for_intent_not_used_in_story(
    caplog: LogCaptureFixture, validator_under_test: Validator
):
    caplog.clear()
    with pytest.warns(UserWarning) as record:
        validator_under_test.verify_intents_in_stories(False)

    assert "The intent 'goodbye' is not used in any story or rule." in (
        m.message.args[0] for m in record
    )


def test_verify_logging_message_for_unused_utterance(
    caplog: LogCaptureFixture, validator_under_test: Validator
):
    caplog.clear()
    with pytest.warns(UserWarning) as record:
        validator_under_test.verify_utterances_in_stories(False)

    assert "The utterance 'utter_chatter' is not used in any story or rule." in (
        m.message.args[0] for m in record
    )


def test_verify_logging_message_for_repetition_in_intents(caplog, nlu_data_path: Text):
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path],
    )
    validator = Validator.from_importer(importer)
    caplog.clear()  # clear caplog to avoid counting earlier debug messages
    with pytest.warns(UserWarning) as record:
        validator.verify_example_repetition_in_intents(False)
    assert len(record) == 1
    assert "You should fix that conflict " in record[0].message.args[0]


def test_early_exit_on_invalid_domain():
    domain_path = "data/test_domains/duplicate_intents.yml"

    importer = RasaFileImporter(domain_path=domain_path)
    with pytest.warns(UserWarning) as record:
        validator = Validator.from_importer(importer)
    validator.verify_domain_validity()

    # two for non-unique domains, two for auto-fill removal
    assert len(record) == 4
    assert any(
        [
            f"Loading domain from '{domain_path}' failed. Using empty domain. "
            "Error: 'Intents are not unique! Found multiple intents with name(s) "
            "['default', 'goodbye']. Either rename or remove the duplicate ones.'"
            in warning.message.args[0]
            for warning in record
        ]
    )
    assert record[0].message.args[0] == record[2].message.args[0]
    assert record[1].message.args[0] == record[3].message.args[0]


def test_verify_there_is_not_example_repetition_in_intents():
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml",
        training_data_paths=["examples/knowledgebasebot/data/nlu.yml"],
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_example_repetition_in_intents(False)


def test_verify_actions_in_stories_not_in_domain(tmp_path: Path, domain_path: Text):
    story_file_name = tmp_path / "stories.yml"
    story_file_name.write_text(
        """
        version: "3.0"
        stories:
        - story: story path 1
          steps:
          - intent: greet
          - action: action_test_1
        """
    )

    importer = RasaFileImporter(
        domain_path=domain_path, training_data_paths=[story_file_name],
    )
    validator = Validator.from_importer(importer)
    with pytest.warns(UserWarning) as warning:
        validity = validator.verify_actions_in_stories_rules()
        assert validity is False

    assert (
        "The action 'action_test_1' is used in the 'story path 1' block, "
        "but it is not listed in the domain file." in warning[0].message.args[0]
    )


def test_verify_actions_in_rules_not_in_domain(tmp_path: Path, domain_path: Text):
    rules_file_name = tmp_path / "rules.yml"
    rules_file_name.write_text(
        """
        version: "3.0"
        rules:
        - rule: rule path 1
          steps:
          - intent: goodbye
          - action: action_test_2
        """
    )
    importer = RasaFileImporter(
        domain_path=domain_path, training_data_paths=[rules_file_name],
    )
    validator = Validator.from_importer(importer)
    with pytest.warns(UserWarning) as warning:
        validity = validator.verify_actions_in_stories_rules()
        assert validity is False

    assert (
        "The action 'action_test_2' is used in the 'rule path 1' block, "
        "but it is not listed in the domain file." in warning[0].message.args[0]
    )


def test_verify_form_slots_invalid_domain(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        """
        version: "3.0"
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
    with pytest.warns(UserWarning) as w:
        validity = validator.verify_form_slots()
        assert validity is False

    assert (
        w[0].message.args[0] == "The form slot 'last_nam' in form 'name_form' "
        "is not present in the domain slots."
        "Please add the correct slot or check for typos."
    )


def test_response_selector_responses_in_domain_no_errors():
    importer = RasaFileImporter(
        config_file="data/test_config/config_defaults.yml",
        domain_path="data/test_domains/response_selector_responses_in_domain.yml",
        training_data_paths=[
            "data/test_yaml_stories/test_base_retrieval_intent_story.yml"
        ],
        training_type=TrainingType.CORE,
    )
    validator = Validator.from_importer(importer)
    assert validator.verify_utterances_in_stories(ignore_warnings=True)


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
        """
        version: "3.0"
        intents:
        - greet
        actions:
        - action_greet
        """
    )
    file_name = tmp_path / f"{file_name}.yml"
    file_name.write_text(
        f"""
        version: "3.0"
        {file_name}:
        - {data_type}: test path
          steps:
          - intent: greet
          - action: action_greet
        """
    )
    importer = RasaFileImporter(domain_path=domain, training_data_paths=[file_name],)
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
        """
        version: "3.0"
        intents:
        - greet
        """
    )
    file_name = tmp_path / f"{file_name}.yml"
    file_name.write_text(
        f"""
            version: "3.0"
            {file_name}:
            - {data_type}: test path
              steps:
              - intent: greet
              - action: action_restart
            """
    )
    importer = RasaFileImporter(domain_path=domain, training_data_paths=[file_name],)
    validator = Validator.from_importer(importer)
    assert validator.verify_actions_in_stories_rules()


def test_valid_form_slots_in_domain(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        """
        version: "3.0"
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


def test_verify_slot_mappings_mapping_active_loop_not_in_forms(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    slot_name = "some_slot"
    domain.write_text(
        f"""
        version: "3.0"
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
    with pytest.warns(
        UserWarning,
        match=r"Slot 'some_slot' has a mapping condition "
        r"for form 'som_form' which is not listed "
        r"in domain forms.*",
    ):
        assert not validator.verify_slot_mappings()


def test_verify_slot_mappings_from_trigger_intent_mapping_slot_not_in_forms(
    tmp_path: Path,
):
    domain = tmp_path / "domain.yml"
    slot_name = "started_booking_form"
    domain.write_text(
        f"""
        version: "3.0"
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
    with pytest.warns(
        UserWarning,
        match=f"Slot '{slot_name}' has a 'from_trigger_intent' mapping, "
        f"but it's not listed in any form 'required_slots'.",
    ):
        assert not validator.verify_slot_mappings()


def test_verify_slot_mappings_slot_with_mapping_conditions_not_in_form(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        """
        version: "3.0"
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
    with pytest.warns(
        UserWarning,
        match=r"Slot 'location' has a mapping condition for form 'booking_form', "
        r"but it's not present in 'booking_form' form's 'required_slots'.*",
    ):
        assert not validator.verify_slot_mappings()


def test_verify_slot_mappings_valid(tmp_path: Path):
    domain = tmp_path / "domain.yml"
    domain.write_text(
        """
        version: "3.0"
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
