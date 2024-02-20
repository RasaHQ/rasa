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
        ("PERSON", "Mein Name ist Julia."),
        ("PHONE_NUMBER", "Rufen Sie mich an unter 020 123 4567."),
        ("EMAIL_ADDRESS", "Senden Sie mir eine E-Mail an test@test.com"),
        (
            "IBAN_CODE",
            "Sie können Geld an die IBAN ES79 2100 0813 6101 2345 6789 überweisen.",
        ),
        ("CREDIT_CARD", "Verwenden Sie diese Kreditkartennummer 4916741327614057."),
        ("DATE_TIME", "Mein Geburtsdatum ist der 01.01.1990."),
        ("IP_ADDRESS", "Wechseln Sie zu 127.0.0.1"),
        ("URL", "Ich muss Informationen auf www.test.com finden"),
        ("LOCATION", "Meine Adresse ist Hauptstraße 123, Berlin"),
    ],
)
def test_anonymization_rule_executor_in_non_default_german_language(
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
        language="de",
        models="de_core_news_lg",
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
