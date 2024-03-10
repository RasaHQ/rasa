from typing import Dict, List, Text

import pytest

import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message


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
                    "pattern": "(\\btacos\\b|\\bbeef\\b|\\bmapo\\ "
                    "tofu\\b|\\bburrito\\b|\\blettuce\\ wrap\\b)",
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


@pytest.mark.parametrize(
    "entity, regex_features, expected_patterns",
    [
        ("", {}, []),
        (
            "zipcode",
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            [{"name": "zipcode", "pattern": "[0-9]{5}"}],
        ),
        ("entity", {"name": "zipcode", "pattern": "[0-9]{5}"}, []),
    ],
)
def test_extract_patterns_use_only_entities_regexes(
    entity: Text, regex_features: Dict[Text, Text], expected_patterns: Dict[Text, Text]
):
    training_data = TrainingData()
    if entity:
        training_data.training_examples = [
            Message(
                data={
                    "text": "text",
                    "intent": "greet",
                    "entities": [{"entity": entity, "value": "text"}],
                }
            )
        ]
    if regex_features:
        training_data.regex_features = [regex_features]

    actual_patterns = pattern_utils.extract_patterns(
        training_data, use_only_entities=True
    )

    assert actual_patterns == expected_patterns


@pytest.mark.parametrize(
    "entity, lookup_tables, expected_patterns",
    [
        ("", {}, []),
        (
            "person",
            {"name": "person", "elements": ["Max", "John"]},
            [{"name": "person", "pattern": "(\\bMax\\b|\\bJohn\\b)"}],
        ),
        ("entity", {"name": "person", "elements": ["Max", "John"]}, []),
    ],
)
def test_extract_patterns_use_only_entities_lookup_tables(
    entity: Text, lookup_tables: Dict[Text, Text], expected_patterns: Dict[Text, Text]
):
    training_data = TrainingData()
    if entity:
        training_data.training_examples = [
            Message(
                data={
                    "text": "text",
                    "intent": "greet",
                    "entities": [{"entity": entity, "value": "text"}],
                }
            )
        ]
    if lookup_tables:
        training_data.lookup_tables = [lookup_tables]

    actual_patterns = pattern_utils.extract_patterns(
        training_data, use_only_entities=True
    )

    assert actual_patterns == expected_patterns


@pytest.mark.parametrize(
    "lookup_tables, regex_features, use_lookup_tables, "
    "use_regex_features, expected_patterns",
    [
        ({"name": "person", "elements": ["Max", "John"]}, {}, False, True, []),
        ({}, {}, True, True, []),
        ({}, {"name": "zipcode", "pattern": "[0-9]{5}"}, True, False, []),
        (
            {"name": "person", "elements": ["Max", "John"]},
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            False,
            False,
            [],
        ),
        (
            {"name": "person", "elements": ["Max", "John"]},
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            True,
            False,
            [{"name": "person", "pattern": "(\\bMax\\b|\\bJohn\\b)"}],
        ),
        (
            {"name": "person", "elements": ["Max", "John"]},
            {"name": "zipcode", "pattern": "[0-9]{5}"},
            False,
            True,
            [{"name": "zipcode", "pattern": "[0-9]{5}"}],
        ),
    ],
)
def test_extract_patterns_use_only_lookup_tables_or_regex_features(
    lookup_tables: Dict[Text, List[Text]],
    regex_features: Dict[Text, Text],
    use_lookup_tables: bool,
    use_regex_features: bool,
    expected_patterns: Dict[Text, Text],
):
    training_data = TrainingData()
    if lookup_tables:
        training_data.lookup_tables = [lookup_tables]
    if regex_features:
        training_data.regex_features = [regex_features]

    actual_patterns = pattern_utils.extract_patterns(
        training_data,
        use_lookup_tables=use_lookup_tables,
        use_regexes=use_regex_features,
    )

    assert actual_patterns == expected_patterns


@pytest.mark.parametrize(
    "lookup_tables, regex_features, use_lookup_tables, use_regex_features",
    [
        (
            {"name": "person", "elements": ["Max", "John"]},
            {"name": "zipcode", "pattern": "*[0-9]{5}"},
            True,
            True,
        )
    ],
)
def test_regex_validation(
    lookup_tables: Dict[Text, List[Text]],
    regex_features: Dict[Text, Text],
    use_lookup_tables: bool,
    use_regex_features: bool,
):
    """Tests if exception is raised when regex patterns are invalid."""

    training_data = TrainingData()
    if lookup_tables:
        training_data.lookup_tables = [lookup_tables]
    if regex_features:
        training_data.regex_features = [regex_features]

    with pytest.raises(Exception) as e:
        pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=use_lookup_tables,
            use_regexes=use_regex_features,
        )

    assert "Model training failed." in str(e.value)
    assert "not a valid regex." in str(e.value)
    assert "Please update your nlu training data configuration" in str(e.value)
