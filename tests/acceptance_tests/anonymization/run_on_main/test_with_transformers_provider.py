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


def load_transformers_model() -> None:
    # download huggingface model
    # make sure to have run `pip install transformers` before
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, TFAutoModelForTokenClassification

    transformers_model = "dslim/bert-base-NER"

    snapshot_download(repo_id=transformers_model)

    # Instantiate to make sure it's downloaded during installation and not runtime
    AutoTokenizer.from_pretrained(transformers_model)
    TFAutoModelForTokenClassification.from_pretrained(transformers_model)


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
def test_anonymization_rule_executor_with_transformers_provider(
    entity_type: Text,
    text: Text,
) -> None:
    load_transformers_model()

    rule_list = AnonymizationRuleList(
        id="test",
        rule_list=[
            AnonymizationRule(
                entity_name=entity_type,
                substitution="faker",
            ),
        ],
        language="en",
        model_provider="transformers",
        models={"spacy": "en_core_web_lg", "transformers": "dslim/bert-base-NER"},
    )

    anonymization_rule_executor = AnonymizationRuleExecutor(rule_list)
    anonymized_text = anonymization_rule_executor.run(text)

    assert anonymized_text != text
