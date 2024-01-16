import logging
import re
from typing import Text
from unittest.mock import Mock

import presidio_analyzer
import pytest
from pytest import LogCaptureFixture, MonkeyPatch
from rasa.shared.exceptions import RasaException

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationAnalyzer,
    AnonymizationRule,
    AnonymizationRuleExecutor,
    AnonymizationRuleList,
)


@pytest.fixture(scope="session")
def mixed_anonymization_rule_list() -> AnonymizationRuleList:
    return AnonymizationRuleList(
        id="test",
        rule_list=[
            AnonymizationRule(
                entity_name="EMAIL_ADDRESS",
                substitution="mask",
            ),
            AnonymizationRule(
                entity_name="PERSON",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="PHONE_NUMBER",
                substitution="mask",
            ),
            AnonymizationRule(
                entity_name="LOCATION",
                substitution="text",
                value="Berlin",
            ),
        ],
    )


@pytest.fixture(scope="session")
def anonymization_rule_executor(
    mixed_anonymization_rule_list: AnonymizationRuleList,
) -> AnonymizationRuleExecutor:
    return AnonymizationRuleExecutor(mixed_anonymization_rule_list)


@pytest.fixture(scope="session")
def faker_anonymization_rule_list() -> AnonymizationRuleList:
    return AnonymizationRuleList(
        id="test-faker",
        rule_list=[
            AnonymizationRule(
                entity_name="PERSON",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="PHONE_NUMBER",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="EMAIL_ADDRESS",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="CREDIT_CARD",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="IBAN_CODE",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="DATE_TIME",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="IP_ADDRESS",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="URL",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="LOCATION",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="US_ITIN",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="US_PASSPORT",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="US_SSN",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="ES_NIF",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="IT_VAT_CODE",
                substitution="faker",
            ),
        ],
    )


@pytest.fixture(scope="session")
def faker_anonymization_rule_executor(
    faker_anonymization_rule_list: AnonymizationRuleList,
) -> AnonymizationRuleExecutor:
    return AnonymizationRuleExecutor(faker_anonymization_rule_list)


def test_anonymization_rule_executor_anonymize_one_entity(
    anonymization_rule_executor: AnonymizationRuleExecutor,
) -> None:
    text = "My email is test@test.com"

    anonymized_text = anonymization_rule_executor.run(text)
    assert anonymized_text == "My email is *************"


@pytest.mark.parametrize(
    "name, phone_number",
    [
        ("John Doe", "020 123 4567"),
        ("Michael O'Connor", "+447976543210"),
        ("Maria Sanchez-Romero", "+34 669 232 374"),
    ],
)
def test_anonymization_rule_executor_multiple_entities(
    anonymization_rule_executor: AnonymizationRuleExecutor,
    name: Text,
    phone_number: Text,
) -> None:
    text = f"My name is {name} and my phone number is {phone_number}."

    anonymized_text = anonymization_rule_executor.run(text)

    assert name not in anonymized_text
    assert phone_number not in anonymized_text

    masked_phone_number = "*" * len(phone_number)
    assert masked_phone_number in anonymized_text

    match = re.match(r"My name is [a-zA-Z'\-.]+\s[a-zA-Z'\-.]+", anonymized_text)
    assert match is not None


def test_anonymization_rule_executor_no_entities(
    anonymization_rule_executor: AnonymizationRuleExecutor,
) -> None:

    text = "Navigate to www.test-domain.com."

    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text == text


def test_anonymization_rule_executor_text_operator(
    anonymization_rule_executor: AnonymizationRuleExecutor,
) -> None:
    text = "I live in Bordeaux."

    anonymized_text = anonymization_rule_executor.run(text)
    assert anonymized_text == "I live in Berlin."


@pytest.mark.parametrize(
    "text",
    [
        "My name is Jane Doe.",
        "My phone number is 020 123 4567.",
        "My email is test@myemail.com",
        "My credit card number is 4916741327614057.",
        "My IBAN code is DE89370400440532013000.",
        "My date of birth is 01/01/1990.",
        "My IP address is 127.0.0.1",
        "I need to search for information on https://www.test.com",
        "I live in Berlin.",
        "My ITIN is 949-71-6890",
        "My US passport number: 604876475",
        "My US SSN is 865-50-6891",
        "My Spanish NIF is 32807127L",
        "My Italian VAT code is 07643520567",
    ],
)
def test_anonymization_rule_executor_all_faker_entities(
    faker_anonymization_rule_executor: AnonymizationRuleExecutor,
    text: Text,
) -> None:
    anonymized_text = faker_anonymization_rule_executor.run(text)

    assert text != anonymized_text


@pytest.mark.parametrize(
    "text, rule_list",
    [
        (
            "I am German.",
            AnonymizationRuleList(
                id="test",
                rule_list=[
                    AnonymizationRule(
                        entity_name="NRP",
                        substitution="faker",
                    ),
                ],
            ),
        ),
        (
            "My bitcoin wallet address is 3FZbgi29cpjq2GjdwV8eyHuJJnkLtktZc5.",
            AnonymizationRuleList(
                id="test",
                rule_list=[
                    AnonymizationRule(
                        entity_name="CRYPTO",
                        substitution="faker",
                    ),
                ],
            ),
        ),
        (
            "My driver license number is A0002144",
            AnonymizationRuleList(
                id="test",
                rule_list=[
                    AnonymizationRule(
                        entity_name="US_DRIVER_LICENSE",
                        substitution="faker",
                    ),
                ],
            ),
        ),
    ],
)
def test_anonymization_rule_executor_defaults_with_unsupported_faker_entity(
    text: Text,
    rule_list: AnonymizationRuleList,
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anon_text = anonymization_rule_executor.run(text)

    assert (
        f"Unsupported faker entity: {rule_list.rule_list[0].entity_name}" in caplog.text
    )
    assert text != anon_text
    # The text should be anonymized with mask substitution
    assert "*" in anon_text


def test_anonymization_analyzer_is_singleton(
    mixed_anonymization_rule_list: AnonymizationRuleList,
) -> None:
    analyzer_1 = AnonymizationAnalyzer(mixed_anonymization_rule_list)
    analyzer_2 = AnonymizationAnalyzer(mixed_anonymization_rule_list)

    assert analyzer_1 is analyzer_2
    assert analyzer_1.presidio_analyzer_engine is analyzer_2.presidio_analyzer_engine


def test_anonymization_analyzer_raises_exception_mocked_stanza_engine(
    monkeypatch: MonkeyPatch,
) -> None:
    AnonymizationAnalyzer.clear()

    monkeypatch.setattr(
        presidio_analyzer.nlp_engine.stanza_nlp_engine.StanzaNlpEngine,
        "__init__",
        Mock(side_effect=ImportError()),
    )
    anonymization_rule_list = AnonymizationRuleList(
        id="some_id",
        model_provider="stanza",
        models="en1",
        rule_list=[
            AnonymizationRule(
                entity_name="PERSON",
                substitution="mask",
            ),
        ],
    )
    with pytest.raises(RasaException, match="Failed to load Presidio nlp engine."):
        AnonymizationAnalyzer(anonymization_rule_list)


def test_anonymization_analyzer_raises_exception_mocked_transformers_engine(
    monkeypatch: MonkeyPatch,
) -> None:
    AnonymizationAnalyzer.clear()

    monkeypatch.setattr(
        presidio_analyzer.nlp_engine.transformers_nlp_engine.TransformersNlpEngine,
        "__init__",
        Mock(side_effect=OSError()),
    )
    anonymization_rule_list = AnonymizationRuleList(
        id="some_id",
        model_provider="transformers",
        models="model_not_existing",
        rule_list=[
            AnonymizationRule(
                entity_name="PERSON",
                substitution="mask",
            ),
        ],
    )

    with pytest.raises(RasaException, match="Failed to load Presidio nlp engine."):
        AnonymizationAnalyzer(anonymization_rule_list)
