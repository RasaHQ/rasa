import pytest
from rasa.shared.exceptions import RasaException

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationAnalyzer,
    AnonymizationRule,
    AnonymizationRuleList,
)


def test_anonymization_analyzer_raises_exception_with_invalid_spacy_model() -> None:
    AnonymizationAnalyzer.clear()

    anonymization_rule_list = AnonymizationRuleList(
        id="some_id",
        model_provider="spacy",
        models="en_core_not_existing",
        rule_list=[
            AnonymizationRule(
                entity_name="PERSON",
                substitution="mask",
            ),
        ],
    )
    with pytest.raises(RasaException, match="Failed to load Presidio nlp engine."):
        AnonymizationAnalyzer(anonymization_rule_list)
