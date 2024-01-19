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


def load_stanza_model() -> None:
    # make sure to run `pip install stanza` and
    # `pip install spacy-stanza` before

    import stanza

    stanza.download("en")


@pytest.mark.parametrize(
    "entity_type, text",
    [
        ("PERSON", "My name is Julia."),
        ("PHONE_NUMBER", "Give me a call on 020 123 4567."),
        ("EMAIL_ADDRESS", "Send me an email on test@test.com"),
        (
            "IBAN_CODE",
            "Please transfer money to IBAN ES79 2100 0813 6101 2345 6789.",
        ),
        ("CREDIT_CARD", "Use this credit card number 4916741327614057."),
        ("DATE_TIME", "My birthdate is 01.01.1990."),
        ("IP_ADDRESS", "Navigate to 127.0.0.1"),
        ("URL", "I'm trying to find information on www.test.com."),
        ("LOCATION", "My address is 123 Main Street, London"),
    ],
)
def test_anonymization_rule_executor_with_stanza_provider(
    entity_type: Text,
    text: Text,
) -> None:
    load_stanza_model()

    rule_list = AnonymizationRuleList(
        id="test",
        rule_list=[
            AnonymizationRule(
                entity_name=entity_type,
                substitution="faker",
            ),
        ],
        language="en",
        model_provider="stanza",
        models="en",
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
