from typing import Dict, List, Text

import pytest

import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.nlu.training_data import TrainingData


@pytest.mark.parametrize(
    "lookup_tables, regex_features, expected_patterns",
    [
        (
            {"name": "person", "elements": ["Max", "John"]},
            {},
            [{"name": "person", "pattern": "(\\bMax\\b|\\bJohn\\b)"}],
        ),
        ({}, {}, []),
        (
            {},
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            [{"name": "zipcode", "pattern": "[0-9]{5}"}],
        ),
        (
            {"name": "person", "elements": ["Max", "John"]},
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            [
                {"name": "zipcode", "pattern": "[0-9]{5}"},
                {"name": "person", "pattern": "(\\bMax\\b|\\bJohn\\b)"},
            ],
        ),
        (
            {"name": "plates", "elements": "data/test/lookup_tables/plates.txt"},
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            [
                {"name": "zipcode", "pattern": "[0-9]{5}"},
                {
                    "name": "plates",
                    "pattern": "(\\btacos\\b|\\bbeef\\b|\\bmapo\\ tofu\\b|\\bburrito\\b|\\blettuce\\ wrap\\b)",
                },
            ],
        ),
    ],
)
def test_extract_patterns(
    lookup_tables: Dict[Text, List[Text]],
    regex_features: Dict[Text, Text],
    expected_patterns: Dict[Text, Text],
):
    training_data = TrainingData()
    if lookup_tables:
        training_data.lookup_tables = [lookup_tables]
    if regex_features:
        training_data.regex_features = [regex_features]

    actual_patterns = pattern_utils.extract_patterns(training_data)

    assert actual_patterns == expected_patterns
