from typing import Text

import pytest

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationRule,
    AnonymizationRuleExecutor,
    AnonymizationRuleList,
)

# The individual tests in this `tests/integration/anonymization` package were separated
# into different modules because run together in the same pytest session would cause
# the majority to fail apart from the first parameterized test. The first test would
# load a particular language model and the subsequent tests would fail because the
# `AnonymizationAnalyzer` component is a Singleton and would not be able to load a
# different language model again.


@pytest.mark.parametrize(
    "entity_type, text",
    [
        ("PERSON", "Mijn naam is Gérard"),
        ("PHONE_NUMBER", "Bel mij op 020 123 4567."),
        ("EMAIL_ADDRESS", "E-mail mij op test@test.com"),
        (
            "IBAN_CODE",
            "U kunt geld overmaken naar IBAN ES79 2100 0813 6101 2345 6789",
        ),
        ("CREDIT_CARD", "Gebruik dit creditcardnummer 4916741327614057."),
        ("DATE_TIME", "Mijn geboortedatum is 01/01/1990."),
        ("IP_ADDRESS", "Schakel over naar 127.0.0.1"),
        ("URL", "Ik moet informatie vinden op www.test.com"),
        ("LOCATION", "Mijn adres is 1234 Main Street, Amsterdam"),
    ],
)
def test_anonymization_rule_executor_in_non_default_dutch_language(
    entity_type: Text,
    text: Text,
) -> None:
    rule_list = AnonymizationRuleList(
        id="test",
        rule_list=[
            AnonymizationRule(
                entity_name=entity_type,
                substitution="mask",
            ),
        ],
        language="nl",
        models="nl_core_news_lg",
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
