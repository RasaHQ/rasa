from textwrap import dedent
from typing import Any, Dict, Text

import pytest
from rasa.shared.exceptions import RasaException

from rasa.anonymization.utils import (
    extract_anonymization_traits,
    read_endpoint_config,
    validate_anonymization_yaml,
)
from rasa.utils.validation import read_yaml


@pytest.mark.parametrize(
    "filename, endpoint_type",
    [
        ("", "anonymization"),
        ("/unknown/path.yml", "anonymization"),
        ("tests/anonymization/fixtures/endpoints.yml", "stuff"),
        ("tests/anonymization/fixtures/endpoints.yml", ""),
    ],
)
def test_read_endpoint_config_not_found(filename: Text, endpoint_type: Text) -> None:
    result = read_endpoint_config(filename, endpoint_type)

    assert result is None


def test_read_endpoint_config() -> None:
    filename = "tests/anonymization/fixtures/endpoints.yml"
    endpoint_type = "anonymization"

    expected = {
        "metadata": {
            "language": "en",
            "model_name": "model_name",
            "model_provider": "spacy",
        },
        "rule_lists": [
            {
                "id": "Rule1",
                "rules": [
                    {"entity": "PERSON", "substitution": "faker"},
                    {"entity": "LOCATION", "substitution": "mask"},
                ],
            },
            {"id": "Rule2", "rules": [{"entity": "CITY", "substitution": "mask"}]},
            {
                "id": "Rule3",
                "rules": [
                    {"entity": "GENDER", "substitution": "text", "value": "female"}
                ],
            },
            {
                "id": "Rule4",
                "rules": [
                    {"entity": "GENDER"},
                    {"entity": "PERSON"},
                ],
            },
        ],
    }

    result = read_endpoint_config(filename, endpoint_type)

    assert result is not None
    assert endpoint_type in result.keys()
    assert expected in result.values()


@pytest.mark.parametrize(
    "filename, endpoint_type, expected",
    [
        (
            "tests/anonymization/fixtures/endpoints.yml",
            "anonymization",
            {
                "enabled": True,
                "metadata": {
                    "language": "en",
                    "model_name": "model_name",
                    "model_provider": "spacy",
                },
                "number_of_rule_lists": 4,
                "number_of_rules": 6,
                "substitutions": {"mask": 2, "faker": 1, "text": 1, "not_defined": 2},
                "entities": ["PERSON", "CITY", "LOCATION", "GENDER"],
            },
        ),
        (  # Empty anonymization section in endpoints.yml -> disabled
            "tests/anonymization/fixtures/endpoints_empty_anonymization.yaml",
            "anonymization",
            {
                "enabled": False,
            },
        ),
        (  # Anonymization section not present in endpoints.yml -> disabled
            "tests/anonymization/fixtures/endpoints_anonymization_not_present.yaml",
            "anonymization",
            {
                "enabled": False,
            },
        ),
    ],
)
def test_extract_anonymization_traits(
    filename: Text,
    endpoint_type: Text,
    expected: Dict[Text, Any],
) -> None:
    result = read_endpoint_config(filename, endpoint_type)

    traits = extract_anonymization_traits(result, endpoint_type)

    assert traits["enabled"] == expected["enabled"]
    # Traits present when anonymization is enabled
    assert traits.get("metadata") == expected.get("metadata")
    assert traits.get("number_of_rule_lists") == expected.get("number_of_rule_lists")
    assert traits.get("number_of_rules") == expected.get("number_of_rules")
    assert traits.get("substitutions") == expected.get("substitutions")
    for entity in traits.get("entities", []):
        assert entity in expected["entities"]


@pytest.mark.parametrize(
    "example",
    [  # duplicate rule ids
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: Rule1
              rules:
                - entity: PERSON
                  substitution: faker
            - id: Rule1
              rules:
                - entity: GENDER
                  substitution: mask
        """
        ),  # rules id is missing
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
              - rules:
                - entity: GENDER
                  substitution: mask
        """
        ),  # rules is missing
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: Rule1
        """
        ),  # rules is empty
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: Rule1
              rules:
        """
        ),  # rule_lists is missing
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
        """
        ),  # rule_lists is empty
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
        """
        ),  # just the key anonymization
        dedent(
            """
        anonymization:
        """
        ),  # model_name is missing
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
          rule_lists:
            - id: Rule1
              rules:
                - entity: PERSON
                  substitution: faker
        """
        ),  # model_provider is missing
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_name: en_core_web_lg
          rule_lists:
            - id: Rule1
              rules:
                - entity: PERSON
                  substitution: faker
        """
        ),  # language is missing
        dedent(
            """
        anonymization:
          metadata:
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: Rule1
              rules:
                - entity: PERSON
                  substitution: faker
        """
        ),  # metadata is missing
        dedent(
            """
        anonymization:
          rule_lists:
            - id: Rule1
              rules:
                - entity: PERSON
                  substitution: faker
        """
        ),  # substitution method does not exist
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: Rule1
              rules:
                - entity: PERSON
                  substitution: fake
        """
        ),  # id is a number
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: 1
              rules:
                - entity: PERSON
                  substitution: faker
        """
        ),
    ],
)
def test_validate_wrong_schema_raises(example: str) -> None:
    yaml_content = read_yaml(example)
    with pytest.raises(RasaException):
        validate_anonymization_yaml(yaml_content)


@pytest.mark.parametrize(
    "example",
    [
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: rule_1
              rules:
                - entity: PERSON
                  substitution: faker
        """
        ),
        dedent(
            """
        anonymization:
          metadata:
            language: en
            model_provider: spacy
            model_name: en_core_web_lg
          rule_lists:
            - id: rule_1
              rules:
                - entity: PERSON
                  substitution: faker
            - id: Rule2
              rules:
                - entity: GENDER
                  substitution: mask
                - entity: LOCATION
        """
        ),
    ],
)
def test_validate_right_yaml(example: str) -> None:
    yaml_content = read_yaml(example)
    try:
        validate_anonymization_yaml(yaml_content)
    except RasaException as exception:
        assert False, f"'validation method raised an unexpected exception: {exception}"
