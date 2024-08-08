import json
import textwrap
from typing import Any, Dict, List, Text, Union

import pytest
from rasa.shared.utils.yaml import (
    YamlValidationException,
    parse_raw_yaml,
    validate_yaml_data_using_schema_with_assertions,
)


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
  - test_case: "test_premium_booking_2"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_multiple_test_cases(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
fixtures:
  - premium:
      - membership_type: premium
  - standard:
      - membership_type: standard

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_global_fixtures(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
fixtures:
  - 2premium:
      - membership_type: premium

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_global_fixture_name_must_not_start_with_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
      - utter: "utter_greet"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_multiple_step_in_test_case(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    fixtures:
      - premium
      - standard
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_fixtures_in_test_case(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    fixtures:
      - 2premium
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_fixture_name_in_test_case_must_not_start_with_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - bot: "Hey! How are you?"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_without_fixtures(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: "some value"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_name_without_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot_no2: "some value"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_name_contains_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - 2my_slot_no: "some value"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_name_must_not_start_with_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: 3
"""
        ),
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: 3.14
"""
        ),
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: -3
"""
        ),
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: -3.14
"""
        ),
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: 0
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_with_number_value(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: null
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_with_value_none(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: "Hi"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_with_string_value(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot: True
"""
        ),
    ],
)
def test_e2e_test_cases_schema_slot_with_bool_value(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_mood_great"
    steps:
      - user: "Hi!"
      - slot_was_set:
            - my_slot
"""
        ),
    ],
)
def test_e2e_test_cases_schema_just_slot_name(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
metadata:
  - user_info:
      language: English
      location: Europe

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_global_metadata(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
metadata:
  - 2device_info:
      os: linux

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_global_metadata_name_must_not_start_with_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
metadata:
  user_info:
      language: English
      location: Europe

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_global_metadata_should_be_a_list_of_dict_not_a_dict(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
metadata:
  - user_info:
      language: English
      location: Europe
  -

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_global_metadata_list_should_not_have_an_empty_item(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
metadata:
  - user_info:

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_global_metadata_list_item_is_not_an_empty_dict(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
metadata:
  - :
    os: linux

test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_global_metadata_list_item_has_a_name(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    metadata: "user_info"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_metadata_in_test_case(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    metadata: "2user_info"
    steps:
      - user: "Hi!"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_metadata_name_in_test_case_must_not_start_with_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
        metadata: "user_info"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_with_metadata_in_user_step(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "test_case_file_content",
    [
        (
            """
test_cases:
  - test_case: "test_premium_booking"
    steps:
      - user: "Hi!"
        metadata: "2user_info"
"""
        ),
    ],
)
def test_e2e_test_cases_schema_metadata_name_in_user_step_must_not_start_with_number(
    test_case_file_content: Text,
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )


def test_e2e_test_case_schema_with_valid_assertions(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
) -> None:
    test_case_file_content = textwrap.dedent("""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - flow_started: transfer_money
       - flow_completed:
           flow_id: transfer_money
           flow_step_id: execute_transfer
       - flow_cancelled:
           flow_id: transfer_money
           flow_step_id: execute_transfer
       - pattern_clarification_contains:
           - list_contacts
           - add_contacts
           - remove_contacts
       - action_executed: execute_transfer
       - slot_was_set:
           - name: recipient
             value: John
           - name: amount
             value: 100
       - slot_was_not_set:
           - name: recipient
           - name: amount
             value: 100
       - bot_uttered:
           utter_name: utter_ask_transfer_money_amount
           buttons:
           - title: "100"
             payload: "/SetSlots(amount=100)"
           - title: "200"
             payload: /SetSlots(amount=200)
           text_matches: "How much money to transfer?"
       - generative_response_is_relevant:
            utter_name: utter_chitchat
            threshold: 0.75
       - generative_response_is_grounded:
            threshold: 0.75
            ground_truth: "International transfers charge a commission fee of x %"
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )
    except YamlValidationException as exc:
        assert False, f"'validate_yaml_content_using_schema' raised an exception {exc}"


@pytest.mark.parametrize(
    "flow_started_value",
    [
        None,
        1,
        1.0,
        [],
        {},
        ["transfer_money"],
        {"flow_id": "transfer_money"},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_flow_started_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    flow_started_value: Any,
) -> None:
    flow_started_value = json.dumps(flow_started_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - flow_started: {flow_started_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "flow_started" in str(exc.value)


@pytest.mark.parametrize(
    "flow_completed_value",
    [
        None,
        "",
        1,
        1.0,
        [],
        {},
        ["transfer_money"],
        {"flow_step_id": "START"},
        {"flow_id": None},
        {"flow_id": 1},
        {"flow_id": 1.0},
        {"flow_id": []},
        {"flow_id": {}},
        {"flow_id": ["transfer_money"]},
        {"flow_id": True},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_flow_completed_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    flow_completed_value: Any,
) -> None:
    flow_completed_value = json.dumps(flow_completed_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - flow_completed: {flow_completed_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "flow_completed" in str(exc.value)


@pytest.mark.parametrize(
    "flow_cancelled_value",
    [
        None,
        "",
        1,
        1.0,
        [],
        {},
        ["transfer_money"],
        {"flow_step_id": "START"},
        {"flow_id": None},
        {"flow_id": 1},
        {"flow_id": 1.0},
        {"flow_id": []},
        {"flow_id": {}},
        {"flow_id": ["transfer_money"]},
        {"flow_id": True},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_flow_cancelled_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    flow_cancelled_value: Any,
) -> None:
    flow_cancelled_value = json.dumps(flow_cancelled_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - flow_cancelled: {flow_cancelled_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "flow_cancelled" in str(exc.value)


@pytest.mark.parametrize(
    "pattern_clarification_value",
    [
        None,
        "",
        1,
        1.0,
        {},
        [None],
        [{}],
        [1],
        [1.0],
        [True],
        [False],
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_pattern_clarification_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    pattern_clarification_value: Any,
) -> None:
    pattern_clarification_value = json.dumps(pattern_clarification_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - pattern_clarification_contains: {pattern_clarification_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "pattern_clarification_contains" in str(exc.value)


@pytest.mark.parametrize(
    "action_executed_value",
    [
        None,
        1,
        1.0,
        [],
        {},
        ["transfer_money"],
        {"flow_id": "transfer_money"},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_action_executed_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    action_executed_value: Any,
) -> None:
    action_executed_value = json.dumps(action_executed_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - action_executed: {action_executed_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "action_executed" in str(exc.value)


@pytest.mark.parametrize(
    "slot_was_set_value",
    [
        None,
        "",
        1,
        1.0,
        {"name": None},
        {"name": 1},
        {"name": 1.0},
        {"name": []},
        {"name": {}},
        {"name": ["recipient"]},
        {"name": True},
        {},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_slow_was_set_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    slot_was_set_value: Any,
) -> None:
    slot_was_set_value = json.dumps(slot_was_set_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - slot_was_set: {slot_was_set_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "slot_was_set" in str(exc.value)


@pytest.mark.parametrize(
    "slot_was_not_set_value",
    [
        None,
        "",
        1,
        1.0,
        {"name": None},
        {"name": 1},
        {"name": 1.0},
        {"name": []},
        {"name": {}},
        {"name": ["recipient"]},
        {"name": True},
        {},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_slow_was_not_set_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    slot_was_not_set_value: Any,
) -> None:
    slot_was_not_set_value = json.dumps(slot_was_not_set_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - slot_was_not_set: {slot_was_not_set_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "slot_was_not_set" in str(exc.value)


@pytest.mark.parametrize(
    "bot_uttered_value",
    [
        None,
        "",
        1,
        1.0,
        {"utter_name": None},
        {"utter_name": 1},
        {"utter_name": 1.0},
        {"utter_name": []},
        {"utter_name": {}},
        {"utter_name": ["recipient"]},
        {"utter_name": True},
        {},
        {"buttons": None},
        {"buttons": 1},
        {"buttons": 1.0},
        {"buttons": []},
        {"buttons": {}},
        {"buttons": ["recipient"]},
        {"buttons": True},
        {"buttons": [{"title": 1, "payload": 1}]},
        {"buttons": [{"title": 1.0, "payload": 1.0}]},
        {"buttons": [{"title": [], "payload": []}]},
        {"buttons": [{"title": {}, "payload": {}}]},
        {"buttons": [{"title": ["recipient"], "payload": ["recipient"]}]},
        {"buttons": [{"title": True, "payload": True}]},
        {"buttons": [{"title": None, "payload": None}]},
        {"buttons": [{"random_key": None, "other_key": None}]},
        {"text_matches": None},
        {"text_matches": 1},
        {"text_matches": 1.0},
        {"text_matches": []},
        {"text_matches": {}},
        {"text_matches": ["recipient"]},
        {"text_matches": True},
        True,
        False,
    ],
)
def test_e2e_test_case_schema_assertion_bot_uttered_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    bot_uttered_value: Any,
) -> None:
    bot_uttered_value = json.dumps(bot_uttered_value)
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - bot_uttered: {bot_uttered_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "bot_uttered" in str(exc.value)


@pytest.mark.parametrize(
    "generative_response_is_relevant_value",
    [
        None,
        "",
        1,
        1.0,
        True,
        False,
        {},
        {"threshold": None},
        {"threshold": []},
        {"threshold": {}},
        {"threshold": True},
        {"utter_name": 1},
        {"utter_name": 1.0},
        {"utter_name": []},
        {"utter_name": {}},
        {"utter_name": None},
        {"utter_name": True},
        {"ground_truth": 1},
        {"ground_truth": 1.0},
        {"ground_truth": []},
        {"ground_truth": {}},
        {"ground_truth": None},
        {"ground_truth": True},
    ],
)
def test_e2e_test_case_schema_assertion_generative_response_relevant_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    generative_response_is_relevant_value: Any,
) -> None:
    generative_response_is_relevant_value = json.dumps(
        generative_response_is_relevant_value
    )
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - generative_response_is_relevant: {generative_response_is_relevant_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "generative_response_is_relevant" in str(exc.value)


@pytest.mark.parametrize(
    "generative_response_is_grounded_value",
    [
        None,
        "",
        1,
        1.0,
        True,
        False,
        {},
        {"threshold": None},
        {"threshold": []},
        {"threshold": {}},
        {"threshold": True},
        {"utter_name": 1},
        {"utter_name": 1.0},
        {"utter_name": []},
        {"utter_name": {}},
        {"utter_name": None},
        {"utter_name": True},
        {"ground_truth": 1},
        {"ground_truth": 1.0},
        {"ground_truth": []},
        {"ground_truth": {}},
        {"ground_truth": None},
        {"ground_truth": True},
    ],
)
def test_e2e_test_case_schema_assertion_generative_response_grounded_invalid_type(
    e2e_schema: Union[List[Any], Dict[Text, Any]],
    generative_response_is_grounded_value: Any,
) -> None:
    generative_response_is_grounded_value = json.dumps(
        generative_response_is_grounded_value
    )
    test_case_file_content = textwrap.dedent(f"""
test_cases:
 - test_case: "test_assertions"
   steps:
    - user: "Hi!"
      assertions:
       - generative_response_is_grounded: {generative_response_is_grounded_value}
""")
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException) as exc:
        validate_yaml_data_using_schema_with_assertions(
            yaml_data=parsed_yaml_content,
            schema_content=e2e_schema,
        )

    validation_message = (
        "Error found. Failed to validate YAML. "
        "Please make sure the file is correct and all mandatory "
        "parameters are specified. Here are the errors found during validation:"
    )
    assert validation_message in str(exc.value)
    assert "generative_response_is_grounded" in str(exc.value)
