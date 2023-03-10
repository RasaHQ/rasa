import textwrap
import warnings
from typing import Text

import pytest
from _pytest.logging import LogCaptureFixture
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION

from rasa.validator import Validator

from rasa.shared.importers.rasa import RasaFileImporter
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

    validator = Validator.from_importer(importer)
    # Since the nlu file actually fails validation,
    # record warnings to make sure that the only raised warning
    # is about the duplicate example 'good afternoon'
    with pytest.warns(UserWarning) as record:
        validator.verify_nlu(ignore_warnings=False)
        assert len(record) == 1
        assert (
            "The example 'good afternoon' was found labeled with multiple different"
            in record[0].message.args[0]
        )


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
    # force validator to not ignore warnings (default is True)
    assert not validator.verify_utterances_in_stories(ignore_warnings=False)


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
    caplog: LogCaptureFixture, validator_under_test: Validator
):
    caplog.clear()
    with pytest.warns(UserWarning) as record:
        # force validator to not ignore warnings (default is True)
        validator_under_test.verify_intents(ignore_warnings=False)

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
        # force validator to not ignore warnings (default is True)
        validator_under_test.verify_intents_in_stories(ignore_warnings=False)

    assert "The intent 'goodbye' is not used in any story or rule." in (
        m.message.args[0] for m in record
    )


def test_verify_logging_message_for_unused_utterance(
    caplog: LogCaptureFixture, validator_under_test: Validator
):
    caplog.clear()
    with pytest.warns(UserWarning) as record:
        # force validator to not ignore warnings (default is True)
        validator_under_test.verify_utterances_in_stories(ignore_warnings=False)

    assert "The utterance 'utter_chatter' is not used in any story or rule." in (
        m.message.args[0] for m in record
    )


def test_verify_logging_message_for_repetition_in_intents(caplog, nlu_data_path: Text):
    # moodbot nlu data already has duplicated example 'good afternoon'
    # for intents greet and goodbye
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml", training_data_paths=[nlu_data_path]
    )
    validator = Validator.from_importer(importer)
    caplog.clear()  # clear caplog to avoid counting earlier debug messages
    with pytest.warns(UserWarning) as record:
        # force validator to not ignore warnings (default is True)
        validator.verify_example_repetition_in_intents(ignore_warnings=False)
    assert len(record) == 1
    assert "You should fix that conflict " in record[0].message.args[0]


def test_early_exit_on_invalid_domain():
    domain_path = "data/test_domains/duplicate_intents.yml"

    importer = RasaFileImporter(domain_path=domain_path)
    with pytest.warns(UserWarning) as record:
        warnings.simplefilter("ignore", DeprecationWarning)
        validator = Validator.from_importer(importer)
    validator.verify_domain_validity()

    # two for non-unique domains, 2 for auto-fill removal
    assert len(record) == 4

    non_unique_warnings = list(
        filter(
            lambda warning: f"Loading domain from '{domain_path}' failed. "
            f"Using empty domain. Error: 'Intents are not unique! "
            f"Found multiple intents with name(s) ['default', 'goodbye']. "
            f"Either rename or remove the duplicate ones.'" in warning.message.args[0],
            record,
        )
    )
    assert len(non_unique_warnings) == 2

    auto_fill_warnings = list(
        filter(
            lambda warning: "Slot auto-fill has been removed in 3.0"
            in warning.message.args[0],
            record,
        )
    )
    assert len(auto_fill_warnings) == 2


def test_verify_there_is_not_example_repetition_in_intents():
    importer = RasaFileImporter(
        domain_path="data/test_moodbot/domain.yml",
        training_data_paths=["examples/knowledgebasebot/data/nlu.yml"],
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_example_repetition_in_intents(ignore_warnings=False)


def test_verify_actions_in_stories_not_in_domain(tmp_path: Path, domain_path: Text):
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
    )
    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert validator.verify_utterances_in_stories(ignore_warnings=False)


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


def test_verify_slot_mappings_mapping_active_loop_not_in_forms(tmp_path: Path):
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
    with pytest.warns(
        UserWarning,
        match=r"Slot 'some_slot' has a mapping condition "
        r"for form 'som_form' which is not listed "
        r"in domain forms.*",
    ):
        assert not validator.verify_slot_mappings()


def test_verify_slot_mappings_slot_with_mapping_conditions_not_in_form(tmp_path: Path):
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
    with pytest.warns(
        UserWarning,
        match=r"Slot 'location' has a mapping condition for form 'booking_form', "
        r"but it's not present in 'booking_form' form's 'required_slots'.*",
    ):
        assert not validator.verify_slot_mappings()


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


def test_verify_utterances_does_not_error_when_no_utterance_template_provided(
    tmp_path: Path, nlu_data_path: Path
):
    story_file_name = tmp_path / "stories.yml"
    with open(story_file_name, "w") as file:
        file.write(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            stories:
            - story: path 1
              steps:
              - intent: greet
              - action: utter_greet
            """
        )
    domain_file_name = tmp_path / "domain.yml"
    with open(domain_file_name, "w") as file:
        file.write(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intents:
              - greet
            actions:
              - utter_greet
            """
        )
    importer = RasaFileImporter(
        config_file="data/test_moodbot/config.yml",
        domain_path=domain_file_name,
        training_data_paths=[story_file_name, nlu_data_path],
    )

    validator = Validator.from_importer(importer)
    # force validator to not ignore warnings (default is True)
    assert not validator.verify_utterances_in_stories(ignore_warnings=False)
    # test whether ignoring warnings actually works
    assert validator.verify_utterances_in_stories(ignore_warnings=True)


@pytest.mark.parametrize(
    "config_file, message",
    [
        (
            "data/test_config/config_defaults.yml",
            "The config file is missing a unique value for "
            "the 'assistant_id' mandatory key.",
        ),
        (
            "data/test_config/config_no_assistant_id.yml",
            "The config file is missing the 'assistant_id' mandatory key.",
        ),
    ],
)
def test_warn_if_config_mandatory_keys_are_not_set_invalid_paths(
    config_file: Text, message: Text
) -> None:
    importer = RasaFileImporter(config_file=config_file)
    validator = Validator.from_importer(importer)

    with pytest.warns(UserWarning, match=message):
        validator.warn_if_config_mandatory_keys_are_not_set()
