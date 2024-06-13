from typing import Any, Dict, List, Text, Union

import pytest
from rasa.shared.utils.yaml import (
    parse_raw_yaml,
    YamlValidationException,
    validate_yaml_content_using_schema,
)

from rasa.cli.e2e_test import read_e2e_test_schema


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
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    try:
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
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
    e2e_schema = read_e2e_test_schema()
    parsed_yaml_content = parse_raw_yaml(test_case_file_content)
    with pytest.raises(YamlValidationException):
        validate_yaml_content_using_schema(
            yaml_content=parsed_yaml_content, schema_content=e2e_schema
        )
