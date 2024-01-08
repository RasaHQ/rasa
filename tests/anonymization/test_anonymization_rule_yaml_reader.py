from typing import Text

import pytest
from rasa.shared.exceptions import RasaException

from rasa.anonymization.anonymisation_rule_yaml_reader import (
    AnonymizationRule,
    AnonymizationRuleList,
    AnonymizationRulesYamlReader,
)

RULES = [
    AnonymizationRuleList(
        id="Rule1",
        rule_list=[
            AnonymizationRule(entity_name="PERSON", substitution="faker", value=None),
            AnonymizationRule(entity_name="LOCATION", substitution="mask", value=None),
        ],
        language="en",
        model_provider="spacy",
        models="model_name",
    ),
    AnonymizationRuleList(
        id="Rule2",
        rule_list=[
            AnonymizationRule(entity_name="CITY", substitution="mask", value=None)
        ],
        language="en",
        model_provider="spacy",
        models="model_name",
    ),
    AnonymizationRuleList(
        id="Rule3",
        rule_list=[
            AnonymizationRule(entity_name="GENDER", substitution="text", value="female")
        ],
        language="en",
        model_provider="spacy",
        models="model_name",
    ),
    AnonymizationRuleList(
        id="Rule4",
        rule_list=[
            AnonymizationRule(entity_name="GENDER", substitution="mask", value=None),
            AnonymizationRule(entity_name="PERSON", substitution="mask", value=None),
        ],
        language="en",
        model_provider="spacy",
        models="model_name",
    ),
]


@pytest.mark.parametrize(
    "filename",
    [
        "tests/anonymization/fixtures/wrong_anonymization1.yml",
        "tests/anonymization/fixtures/wrong_anonymization2.yml",
    ],
)
def test_read_invalid_anonymization_rules_raises(filename: Text) -> None:
    parser = AnonymizationRulesYamlReader(filename)
    with pytest.raises(RasaException) as exception:
        parser.read_anonymization_rules()

    assert "Invalid configuration for `anonymization_rules`" in str(exception.value)


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("tests/anonymization/fixtures/no_file.yml", []),
        ("", []),
        ("tests/anonymization/fixtures/endpoints.yml", RULES),
    ],
)
def test_read_anonymization_rules(
    filename: Text,
    expected: Text,
) -> None:
    parser = AnonymizationRulesYamlReader(filename)
    result = parser.read_anonymization_rules()
    assert result == expected


@pytest.mark.parametrize(
    "lang_code",
    [
        "ca",
        "zh",
        "hr",
        "da",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "it",
        "ja",
        "ko",
        "lt",
        "mk",
        "nb",
        "pl",
        "pt",
        "ro",
        "ru",
        "es",
        "sv",
        "uk",
    ],
)
def test_validate_language(
    lang_code: Text,
) -> None:
    """Test that no exception is raised if the language is valid."""
    parser = AnonymizationRulesYamlReader()
    try:
        parser.validate_language(lang_code)
    except RasaException as exception:
        assert False, f"'validation method raised an unexpected exception: {exception}"


@pytest.mark.parametrize("lang_code", ["jp", "xx"])
def test_validate_language_raises(
    lang_code: Text,
) -> None:
    """Test that an exception is raised if the language is invalid."""
    parser = AnonymizationRulesYamlReader()
    message = (
        f"Provided language code '{lang_code}' is invalid. "
        f"In order to proceed with anonymization, "
        f"please provide a valid ISO 639-2 language code"
    )
    with pytest.raises(RasaException, match=message):
        parser.validate_language(lang_code)
