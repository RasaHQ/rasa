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
        ("PERSON", "Je m'appelle Giulia."),
        ("PHONE_NUMBER", "Appelez-moi au 020 123 4567."),
        ("EMAIL_ADDRESS", "Envoyez-moi un e-mail à test@test.com"),
        (
            "IBAN_CODE",
            "Vous pouvez transférer de l'argent vers IBAN "
            "ES79 2100 0813 6101 2345 6789",
        ),
        ("CREDIT_CARD", "Utilisez ce numéro de carte de crédit 4916741327614057."),
        ("DATE_TIME", "Ma date de naissance est le 01/01/1990."),
        ("IP_ADDRESS", "Passer à 127.0.0.1"),
        ("URL", "J'ai besoin de trouver des informations sur www.test.com"),
        ("LOCATION", "Mon adresse est 123 Grande Rue, Paris"),
    ],
)
def test_anonymization_rule_executor_in_non_default_french_language(
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
        language="fr",
        models="fr_core_news_lg",
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
