import textwrap
from typing import Text
from threading import Thread

import pytest

from pep440_version_utils import Version
from rasa.shared.core.flows.yaml_flows_io import FLOWS_SCHEMA_FILE, YAMLFlowsReader

from rasa.shared.exceptions import YamlException, SchemaValidationError
import rasa.shared.utils.io
import rasa.shared.utils.validation as validation_utils
import rasa.utils.io as io_utils
import rasa.shared.nlu.training_data.schemas.data_schema as schema
from rasa.shared.constants import (
    CONFIG_SCHEMA_FILE,
    DOMAIN_SCHEMA_FILE,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.shared.nlu.training_data.formats.rasa_yaml import NLU_SCHEMA_FILE
from rasa.shared.utils.validation import (
    KEY_TRAINING_DATA_FORMAT_VERSION,
    YamlValidationException,
    validate_yaml_with_jsonschema,
)


@pytest.mark.parametrize(
    "file, schema",
    [
        ("data/test_moodbot/domain.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_config/config_defaults.yml", CONFIG_SCHEMA_FILE),
        ("data/test_config/config_supervised_embeddings.yml", CONFIG_SCHEMA_FILE),
        ("data/test_config/config_crf_custom_features.yml", CONFIG_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema(file, schema):
    # should raise no exception
    validation_utils.validate_yaml_schema(rasa.shared.utils.io.read_file(file), schema)


def test_validate_yaml_schema_with_package_name():
    # should raise no exception
    file = "data/test_moodbot/domain.yml"
    schema = DOMAIN_SCHEMA_FILE
    validation_utils.validate_yaml_schema(
        rasa.shared.utils.io.read_file(file), schema, package_name="rasa"
    )


def test_validate_yaml_schema_with_random_package_name_fails():
    # should raise no exception
    file = "data/test_moodbot/domain.yml"
    schema = DOMAIN_SCHEMA_FILE

    with pytest.raises(ModuleNotFoundError):
        validation_utils.validate_yaml_schema(
            rasa.shared.utils.io.read_file(file), schema, package_name="rasa_foo_bar_42"
        )


@pytest.mark.parametrize(
    "file, schema",
    [
        ("data/test_domains/valid_actions.yml", DOMAIN_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema_actions(file: Text, schema: Text):
    # should raise no exception
    validation_utils.validate_yaml_schema(rasa.shared.utils.io.read_file(file), schema)


@pytest.mark.parametrize(
    "content, schema",
    [
        (
            """
        intents:
            - greet

        entities:
            - name

        responses:
            utter_greet:
                - text: hey there!

        actions:
          - utter_default: {send_domain: 1}
        """,
            DOMAIN_SCHEMA_FILE,
        ),
        (
            """
        intents:
            - greet

        entities:
            - name

        responses:
            utter_greet:
                - text: hey there!

        actions:
          - utter_default: {send_domain: 0}
        """,
            DOMAIN_SCHEMA_FILE,
        ),
        (
            """
        intents:
            - greet

        entities:
            - name

        responses:
            utter_greet:
                - text: hey there!

        actions:
          - utter_default: {send_domain: Ttrue}
        """,
            DOMAIN_SCHEMA_FILE,
        ),
        (
            """
        intents:
            - greet

        entities:
            - name

        responses:
            utter_greet:
                - text: hey there!

        actions:
          - utter_default: {send_domain: ""}
        """,
            DOMAIN_SCHEMA_FILE,
        ),
    ],
)
def test_invalid_send_domain_value_in_actions(content: Text, schema: Text):
    with pytest.raises(validation_utils.YamlValidationException):
        validation_utils.validate_yaml_schema(content, schema)


@pytest.mark.parametrize(
    "file, schema",
    [
        ("data/test_domains/invalid_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_domains/wrong_response_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_domains/wrong_custom_response_format.yml", DOMAIN_SCHEMA_FILE),
        ("data/test_domains/empty_response_format.yml", DOMAIN_SCHEMA_FILE),
    ],
)
def test_validate_yaml_schema_raise_exception(file: Text, schema: Text):
    with pytest.raises(YamlException):
        validation_utils.validate_yaml_schema(
            rasa.shared.utils.io.read_file(file), schema
        )


def test_validate_yaml_schema_raise_exception_null_text():
    domain = f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    responses:
      utter_ask_email:
      - text: What is your email ID?
      utter_ask_name:
      - text: null
    """
    with pytest.raises(validation_utils.YamlValidationException) as e:
        validation_utils.validate_yaml_schema(domain, DOMAIN_SCHEMA_FILE)

    assert (
        "Missing 'text' or 'custom' key in response or null 'text' value in response."
        in str(e.value)
    )


def test_validate_yaml_schema_raise_exception_extra_hyphen_for_image():
    domain = f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        responses:
          utter_cheer_up:
          - image: https://i.imgur.com/nGF1K8f.jpg
          - text: Here is something to cheer you up
        """
    with pytest.raises(validation_utils.YamlValidationException) as e:
        validation_utils.validate_yaml_schema(domain, DOMAIN_SCHEMA_FILE)

    assert (
        "Missing 'text' or 'custom' key in response or null 'text' value in response."
        in str(e.value)
    )


def test_example_training_data_is_valid():
    demo_json = "data/examples/rasa/demo-rasa.json"
    data = rasa.shared.utils.io.read_json_file(demo_json)
    validation_utils.validate_training_data(data, schema.rasa_nlu_data_schema())


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"wrong_top_level": []},
        ["this is not a toplevel dict"],
        {
            "rasa_nlu_data": {
                "common_examples": [
                    {
                        "text": "mytext",
                        "entities": [{"start": "INVALID", "end": 0, "entity": "x"}],
                    }
                ]
            }
        },
    ],
)
def test_validate_training_data_is_throwing_exceptions(invalid_data):
    with pytest.raises(SchemaValidationError):
        validation_utils.validate_training_data(
            invalid_data, schema.rasa_nlu_data_schema()
        )


def test_url_data_format():
    data = """
    {
      "rasa_nlu_data": {
        "entity_synonyms": [
          {
            "value": "nyc",
            "synonyms": ["New York City", "nyc", "the big apple"]
          }
        ],
        "common_examples" : [
          {
            "text": "show me flights to New York City",
            "intent": "unk",
            "entities": [
              {
                "entity": "destination",
                "start": 19,
                "end": 32,
                "value": "NYC"
              }
            ]
          }
        ]
      }
    }"""
    fname = io_utils.create_temporary_file(
        data.encode(rasa.shared.utils.io.DEFAULT_ENCODING),
        suffix="_tmp_training_data.json",
        mode="w+b",
    )
    data = rasa.shared.utils.io.read_json_file(fname)
    assert data is not None
    validation_utils.validate_training_data(data, schema.rasa_nlu_data_schema())


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"group": "a", "role": "c", "value": "text"},
        ["this is not a toplevel dict"],
        {"entity": 1, "role": "c", "value": "text"},
        {"entity": "e", "role": None, "value": "text"},
    ],
)
def test_validate_entity_dict_is_throwing_exceptions(invalid_data):
    with pytest.raises(SchemaValidationError):
        validation_utils.validate_training_data(
            invalid_data, schema.entity_dict_schema()
        )


@pytest.mark.parametrize(
    "data",
    [
        {"entity": "e", "group": "a", "role": "c", "value": "text"},
        {"entity": "e"},
        {"entity": "e", "value": "text"},
        {"entity": "e", "group": "a"},
        {"entity": "e", "role": "c"},
        {"entity": "e", "role": "c", "value": "text"},
        {"entity": "e", "group": "a", "value": "text"},
        {"entity": "e", "group": "a", "role": "c"},
        {"entity": "e", "value": 3},
        {"entity": "e", "value": "3"},
    ],
)
def test_entity_dict_is_valid(data):
    validation_utils.validate_training_data(data, schema.entity_dict_schema())


async def test_future_training_data_format_version_not_compatible():

    next_minor = str(Version(LATEST_TRAINING_DATA_FORMAT_VERSION).next_minor())

    incompatible_version = {KEY_TRAINING_DATA_FORMAT_VERSION: next_minor}

    with pytest.warns(UserWarning):
        assert not validation_utils.validate_training_data_format_version(
            incompatible_version, ""
        )


async def test_compatible_training_data_format_version():

    prev_major = str(Version("1.0"))

    compatible_version_1 = {KEY_TRAINING_DATA_FORMAT_VERSION: prev_major}
    compatible_version_2 = {
        KEY_TRAINING_DATA_FORMAT_VERSION: LATEST_TRAINING_DATA_FORMAT_VERSION
    }

    for version in [compatible_version_1, compatible_version_2]:
        with pytest.warns(None):
            assert validation_utils.validate_training_data_format_version(version, "")


async def test_invalid_training_data_format_version_warns():

    invalid_version_1 = {KEY_TRAINING_DATA_FORMAT_VERSION: 2.0}
    invalid_version_2 = {KEY_TRAINING_DATA_FORMAT_VERSION: "Rasa"}

    for version in [invalid_version_1, invalid_version_2]:
        with pytest.warns(UserWarning):
            assert validation_utils.validate_training_data_format_version(version, "")


def test_concurrent_schema_validation():
    successful_results = []

    def validate() -> None:
        payload = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - moin
    - hey there
    - let's go
    - hey dude
    - goodmorning
    - goodevening
    - good afternoon
- intent: goodbye
  examples: |
    - good afternoon
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
        """
        rasa.shared.utils.validation.validate_yaml_schema(payload, NLU_SCHEMA_FILE)
        successful_results.append(True)

    threads = []
    for i in range(10):
        threads.append(Thread(target=validate))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert len(successful_results) == len(threads)


@pytest.mark.parametrize(
    "flow_yaml",
    [
        """flows:
  replace_eligible_card:
    description: Never predict StartFlow for this flow, users are not able to trigger.
    name: replace eligible card
    steps:
      - collect: replacement_reason
        next:
          - if: replacement_reason == "lost"
            then:
              - collect: was_card_used_fraudulently
                ask_before_filling: true
                next:
                  - if: was_card_used_fraudulently
                    then:
                      - action: utter_report_fraud
                        next: END
                  - else: start_replacement
          - if: "replacement_reason == 'damaged'"
            then: start_replacement
          - else:
            - action: utter_unknown_replacement_reason_handover
              next: END
      - id: start_replacement
        action: utter_will_cancel_and_send_new
      - action: utter_new_card_has_been_ordered""",
        """flows:
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
    """,
        f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
flows:
    transfer_money:
        description: This flow lets users send money.
        name: transfer money
        steps:
        - id: "ask_recipient"
          collect: transfer_recipient
          next: "ask_amount"
        - id: "ask_amount"
          collect: transfer_amount
          next: "execute_transfer"
        - id: "execute_transfer"
          action: action_transfer_money""",
        """flows:
  setup_recurrent_payment:
    name: setup recurrent payment
    steps:
      - collect: recurrent_payment_type
        rejections:
          - if: not ({"direct debit" "standing order"} contains recurrent_payment_type)
            utter: utter_invalid_recurrent_payment_type
        description: the type of payment
      - collect: recurrent_payment_recipient
        utter: utter_ask_recipient
        description: the name of a person
      - collect: recurrent_payment_amount_of_money
        description: the amount of money without any currency designation
      - collect: recurrent_payment_frequency
        description: the frequency of the payment
        rejections:
          - if: not ({"monthly" "yearly"} contains recurrent_payment_frequency)
            utter: utter_invalid_recurrent_payment_frequency
      - collect: recurrent_payment_start_date
        description: the start date of the payment
      - collect: recurrent_payment_end_date
        description: the end date of the payment
        rejections:
          - if: recurrent_payment_end_date < recurrent_payment_start_date
            utter: utter_invalid_recurrent_payment_end_date
      - collect: recurrent_payment_confirmation
        description: accepts True or False
        ask_before_filling: true
        next:
          - if: not recurrent_payment_confirmation
            then:
              - action: utter_payment_cancelled
                next: END
          - else: "execute_payment"
      - id: "execute_payment"
        action: action_execute_recurrent_payment
        next:
          - if: setup_recurrent_payment_successful
            then:
              - action: utter_payment_complete
                next: END
          - else: "payment_failed"
      - id: "payment_failed"
        action: utter_payment_failed
      - action: utter_failed_payment_handover
      - action: utter_failed_handoff""",
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              set_slots:
                - foo: bar
              next: "2"
            - id: "2"
              action: action_listen
              next: "1"
              """,
        """
        flows:
          test_flow:
            description: Test flow
            steps:
              - id: "1"
                action: action_xyz
                next: "2"
              - id: "2"
                action: utter_ask_name""",
    ],
)
def test_flow_validation_pass(flow_yaml: str) -> None:
    # test fails if exception is raised
    validate_yaml_with_jsonschema(
        flow_yaml, FLOWS_SCHEMA_FILE, humanize_error=YAMLFlowsReader.humanize_flow_error
    )


def validate_and_return_error_msg(flow_yaml: str) -> str:
    with pytest.raises(YamlValidationException) as e:
        rasa.shared.utils.validation.validate_yaml_with_jsonschema(
            flow_yaml,
            FLOWS_SCHEMA_FILE,
            humanize_error=YAMLFlowsReader.humanize_flow_error,
        )
    return str(e.value)


def test_next_without_then():
    flow = textwrap.dedent(
        """
      flows:
        test:
          steps:
            - action: xyz
              next:
              - if: xyz
    """
    )
    assert (
        "Not a valid 'next' definition. Expected else block or if-then block."
        in validate_and_return_error_msg(flow)
    )


def test_flow_without_content():
    flow = textwrap.dedent(
        """
      flows:
    """
    )
    assert (
        "Found `None` but expected a dictionary of flows."
        in validate_and_return_error_msg(flow)
    )


def test_flow_without_steps():
    flow = textwrap.dedent(
        """
      flows:
        test:
          name: test
          steps:
    """
    )
    assert (
        "Found `None` but expected a list of steps."
        in validate_and_return_error_msg(flow)
    )


def test_flow_with_array_instead_of_object():
    flow = textwrap.dedent(
        """
      flows:
         test:
         - id: test
    """
    )
    assert (
        "Found a list but expected a dictionary with flow properties."
        in validate_and_return_error_msg(flow)
    )


def test_flow_with_boolean_as_next():
    flow = textwrap.dedent(
        """
      flows:
        test:
          name: test
          steps:
            - collect: confirm_correct_card
              ask_before_filling: true
              next:
                - if: "confirm_correct_card"
                  then:
                    - link: "replace_eligible_card"
                - else:
                    - action: utter_relevant_card_not_linked
                      next: True
    """
    )
    assert (
        "Not a valid 'next' definition. Expected list of conditions or step id."
        in validate_and_return_error_msg(flow)
    )


def test_flow_with_a_step_without_a_type():
    flow = textwrap.dedent(
        """
      flows:
        test:
          name: test
          steps:
            - ask_before_filling: true
              next:
                - if: "confirm_correct_card"
                  then:
                    - link: "replace_eligible_card"
                - else:
                    - action: utter_relevant_card_not_linked
                      next: END
    """
    )
    expected_error = (
        "Not a valid 'steps' definition. Expected action step or collect "
        "step or link step or slot set step."
    )
    assert expected_error in validate_and_return_error_msg(flow)


def test_flow_with_a_step_with_ambiguous_type():
    flow = textwrap.dedent(
        """
      flows:
        test:
          steps:
            - collect: confirm_correct_card
              action: utter_xyz
              ask_before_filling: true
    """
    )
    expected_error = (
        "Additional properties are not allowed "
        "('ask_before_filling', 'collect' were unexpected)"
    )
    assert expected_error in validate_and_return_error_msg(flow)


def test_flow_random_unexpected_property_on_action_step():
    flow = textwrap.dedent(
        """
      flows:
        test:
          steps:
            - action: utter_xyz
              random_xyz: true
              next: END
    """
    )
    assert (
        "Additional properties are not allowed ('random_xyz' was unexpected)"
        in validate_and_return_error_msg(flow)
    )


def test_flow_random_unexpected_property_on_collect():
    flow = textwrap.dedent(
        """
      flows:
        test:
          steps:
            - collect: confirm_correct_card
              random_xyz: utter_xyz
              ask_before_filling: true
    """
    )
    assert (
        "Additional properties are not allowed ('random_xyz' was unexpected)"
        in validate_and_return_error_msg(flow)
    )


def test_flow_random_unexpected_property_on_flow():
    flow = textwrap.dedent(
        """
      flows:
        test:
          random_xyz: True
          steps:
            - action: utter_xyz
              next: id-21312
    """
    )
    assert (
        "Additional properties are not allowed ('random_xyz' was unexpected)"
        in validate_and_return_error_msg(flow)
    )


def test_flow_with_invalid_type_for_action():
    flow = textwrap.dedent(
        """
      flows:
        test:
          steps:
            - action: True
              next: id-2132
    """
    )
    assert "Found `True` but expected a string." in validate_and_return_error_msg(flow)


def test_flow_next_is_not_a_step():
    flow = textwrap.dedent(
        """
      flows:
        test:
          steps:
            - action: xyz
              next:
              - action: utter_xyz
    """
    )
    assert (
        "Not a valid 'next' definition. Expected else block or if-then block."
        in validate_and_return_error_msg(flow)
    )
