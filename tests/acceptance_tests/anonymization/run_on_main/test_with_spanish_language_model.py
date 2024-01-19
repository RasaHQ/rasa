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
        ("PERSON", "Mi nombre es David"),
        ("PHONE_NUMBER", "Mi número de teléfono es 020 123 4567."),
        ("EMAIL_ADDRESS", "Mi correo electrónico es test@test.com"),
        ("IBAN_CODE", "Mi IBAN es ES79 2100 0813 6101 2345 6789."),
        ("CREDIT_CARD", "Mi tarjeta de crédito es 4916741327614057."),
        ("DATE_TIME", "Mi cumpleaños es 01/01/1990."),
        ("IP_ADDRESS", "Mi dirección IP es 127.0.0.1"),
        ("URL", "Necesitor buscar information en wwww.test.com"),
        ("LOCATION", "Mi dirección es 1234 Main Street, Anytown, CA 12345."),
    ],
)
def test_anonymization_rule_executor_in_non_default_spanish_language(
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
        language="es",
        models="es_core_news_lg",
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
