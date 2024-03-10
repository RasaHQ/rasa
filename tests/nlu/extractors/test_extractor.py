from typing import Any, Text, Dict, List

import pytest
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION

from rasa.shared.nlu.constants import TEXT, SPLIT_ENTITIES_BY_COMMA
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader


@pytest.mark.parametrize(
    "text, tags, confidences, expected_entities",
    [
        (
            "I am flying from San Fransisco to Amsterdam",
            {
                "entity": ["O", "O", "O", "O", "city", "city", "O", "city"],
                "role": ["O", "O", "O", "O", "from", "from", "O", "to"],
            },
            {
                "entity": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89],
                "role": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89],
            },
            [
                {
                    "entity": "city",
                    "start": 17,
                    "end": 30,
                    "value": "San Fransisco",
                    "role": "from",
                    "confidence_entity": 0.78,
                    "confidence_role": 0.78,
                },
                {
                    "entity": "city",
                    "start": 34,
                    "end": 43,
                    "value": "Amsterdam",
                    "role": "to",
                    "confidence_entity": 0.89,
                    "confidence_role": 0.89,
                },
            ],
        ),
        (
            "I am flying from San Fransisco to Amsterdam",
            {
                "entity": ["O", "O", "O", "O", "B-city", "L-city", "O", "U-city"],
                "role": ["O", "O", "O", "O", "B-from", "L-from", "O", "U-to"],
            },
            {
                "entity": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89],
                "role": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89],
            },
            [
                {
                    "entity": "city",
                    "start": 17,
                    "end": 30,
                    "value": "San Fransisco",
                    "role": "from",
                    "confidence_entity": 0.78,
                    "confidence_role": 0.78,
                },
                {
                    "entity": "city",
                    "start": 34,
                    "end": 43,
                    "value": "Amsterdam",
                    "role": "to",
                    "confidence_entity": 0.89,
                    "confidence_role": 0.89,
                },
            ],
        ),
        (
            "I am flying from San Fransisco to Amsterdam",
            {
                "entity": ["O", "O", "O", "O", "city", "city", "O", "city"],
                "group": ["O", "O", "O", "O", "1", "1", "O", "1"],
            },
            None,
            [
                {
                    "entity": "city",
                    "start": 17,
                    "end": 30,
                    "value": "San Fransisco",
                    "group": "1",
                },
                {
                    "entity": "city",
                    "start": 34,
                    "end": 43,
                    "value": "Amsterdam",
                    "group": "1",
                },
            ],
        ),
        (
            "Amsterdam",
            {"entity": ["city"], "role": ["O"], "group": ["O"]},
            None,
            [{"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"}],
        ),
        (
            "New-York",
            {"entity": ["city", "city"], "role": ["O", "O"], "group": ["O", "O"]},
            None,
            [{"entity": "city", "start": 0, "end": 8, "value": "New-York"}],
        ),
        (
            "Amsterdam, Berlin, and London",
            {
                "entity": ["city", "city", "O", "city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"},
                {"entity": "city", "start": 11, "end": 17, "value": "Berlin"},
                {"entity": "city", "start": 23, "end": 29, "value": "London"},
            ],
        ),
        (
            "Amsterdam Berlin and London",
            {
                "entity": ["U-city", "U-city", "O", "U-city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"},
                {"entity": "city", "start": 10, "end": 16, "value": "Berlin"},
                {"entity": "city", "start": 21, "end": 27, "value": "London"},
            ],
        ),
        (
            "San Fransisco Amsterdam, London",
            {
                "entity": ["B-city", "L-city", "U-city", "U-city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 13, "value": "San Fransisco"},
                {"entity": "city", "start": 14, "end": 23, "value": "Amsterdam"},
                {"entity": "city", "start": 25, "end": 31, "value": "London"},
            ],
        ),
        (
            "New York City Los Angeles and San Diego",
            {
                "entity": [
                    "B-city",
                    "I-city",
                    "L-city",
                    "B-city",
                    "L-city",
                    "O",
                    "B-city",
                    "L-city",
                ],
                "role": ["O", "O", "O", "O", "O", "O", "O", "O"],
                "group": ["O", "O", "O", "O", "O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 13, "value": "New York City"},
                {"entity": "city", "start": 14, "end": 25, "value": "Los Angeles"},
                {"entity": "city", "start": 30, "end": 39, "value": "San Diego"},
            ],
        ),
        (
            "Berlin weather",
            {"entity": ["I-city", "O"], "role": ["O", "O"], "group": ["O", "O"]},
            None,
            [{"entity": "city", "start": 0, "end": 6, "value": "Berlin"}],
        ),
        (
            "New-York",
            {"entity": ["city", "city"], "role": ["O", "O"], "group": ["O", "O"]},
            None,
            [{"entity": "city", "start": 0, "end": 8, "value": "New-York"}],
        ),
        (
            "Amsterdam, Berlin, and London",
            {
                "entity": ["city", "city", "O", "city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"},
                {"entity": "city", "start": 11, "end": 17, "value": "Berlin"},
                {"entity": "city", "start": 23, "end": 29, "value": "London"},
            ],
        ),
        (
            "Amsterdam Berlin and London",
            {
                "entity": ["city", "city", "O", "city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 16, "value": "Amsterdam Berlin"},
                {"entity": "city", "start": 21, "end": 27, "value": "London"},
            ],
        ),
    ],
)
def test_convert_tags_to_entities(
    text: Text,
    tags: Dict[Text, List[Text]],
    confidences: Dict[Text, List[float]],
    expected_entities: List[Dict[Text, Any]],
    whitespace_tokenizer: WhitespaceTokenizer,
):
    extractor = EntityExtractorMixin()

    message = Message(data={TEXT: text})
    tokens = whitespace_tokenizer.tokenize(message, TEXT)

    split_entities_config = {SPLIT_ENTITIES_BY_COMMA: True}
    actual_entities = extractor.convert_predictions_into_entities(
        text, tokens, tags, split_entities_config, confidences
    )
    assert actual_entities == expected_entities


@pytest.mark.parametrize(
    "text, tags, confidences, expected_entities",
    [
        (
            "I live at 22 Powderhall Rd., EH7 4GB, Edinburgh, UK",
            {
                "entity": [
                    "O",
                    "O",
                    "O",
                    "address",
                    "address",
                    "address",
                    "address",
                    "address",
                    "address",
                    "address",
                ]
            },
            {"entity": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89, 1.0, 1.0, 1.0]},
            [
                {
                    "entity": "address",
                    "start": 10,
                    "end": 51,
                    "value": "22 Powderhall Rd., EH7 4GB, Edinburgh, UK",
                    "confidence_entity": 0.78,
                }
            ],
        ),
        (
            "The address is Schönhauser Allee 175, 10119 Berlin, DE",
            {
                "entity": [
                    "O",
                    "O",
                    "O",
                    "address",
                    "address",
                    "address",
                    "address",
                    "address",
                    "address",
                ]
            },
            {"entity": [1.0, 1.0, 1.0, 1.0, 1.0, 0.67, 0.77, 1.0, 0.98]},
            [
                {
                    "entity": "address",
                    "start": 15,
                    "end": 54,
                    "value": "Schönhauser Allee 175, 10119 Berlin, DE",
                    "confidence_entity": 0.67,
                }
            ],
        ),
        (
            "We need to get more of tofu, cauliflower, avocado",
            {
                "entity": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "ingredient",
                    "ingredient",
                    "ingredient",
                ]
            },
            {"entity": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.78]},
            [
                {
                    "entity": "ingredient",
                    "start": 23,
                    "end": 27,
                    "value": "tofu",
                    "confidence_entity": 1.0,
                },
                {
                    "entity": "ingredient",
                    "start": 29,
                    "end": 40,
                    "value": "cauliflower",
                    "confidence_entity": 0.9,
                },
                {
                    "entity": "ingredient",
                    "start": 42,
                    "end": 49,
                    "value": "avocado",
                    "confidence_entity": 0.78,
                },
            ],
        ),
        (
            "So the list of drinks to get is coffee, Club Mate, Ottakringer",
            {
                "entity": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "beverage",
                    "beverage",
                    "beverage",
                    "beverage",
                ]
            },
            {
                "entity": [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.69,
                    0.88,
                    0.84,
                    0.79,
                ]
            },
            [
                {
                    "entity": "beverage",
                    "start": 32,
                    "end": 38,
                    "value": "coffee",
                    "confidence_entity": 0.69,
                },
                {
                    "entity": "beverage",
                    "start": 40,
                    "end": 49,
                    "value": "Club Mate",
                    "confidence_entity": 0.84,
                },
                {
                    "entity": "beverage",
                    "start": 51,
                    "end": 62,
                    "value": "Ottakringer",
                    "confidence_entity": 0.79,
                },
            ],
        ),
    ],
)
def test_split_entities_by_comma(
    text: Text,
    tags: Dict[Text, List[Text]],
    confidences: Dict[Text, List[float]],
    expected_entities: List[Dict[Text, Any]],
    whitespace_tokenizer: WhitespaceTokenizer,
):
    extractor = EntityExtractorMixin()

    message = Message(data={TEXT: text})
    tokens = whitespace_tokenizer.tokenize(message, TEXT)

    split_entities_config = {
        SPLIT_ENTITIES_BY_COMMA: True,
        "address": False,
        "ingredient": True,
    }
    actual_entities = extractor.convert_predictions_into_entities(
        text, tokens, tags, split_entities_config, confidences
    )

    assert actual_entities == expected_entities


@pytest.mark.parametrize(
    "text, warnings",
    [
        (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- intent: test\n"
            "  examples: |\n"
            "    - I want to fly from [Berlin](location) to [ London](location)\n",
            1,
        ),
        (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- intent: test\n"
            "  examples: |\n"
            "    - I want to fly from [Berlin ](location) to [London](location)\n",
            1,
        ),
        (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- intent: test\n"
            "  examples: |\n"
            "    - I want to fly from [Berlin](location) to [London.](location)\n"
            "    - I have nothing to say.\n",
            1,
        ),
        (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- intent: test\n"
            "  examples: |\n"
            "    - I have nothing to say.\n"
            "    - I want to fly from [Berlin](location) to[San Fransisco](location)\n",
            1,
        ),
        (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- intent: test\n"
            "  examples: |\n"
            "    - I want to fly from [Berlin](location) to[San Fransisco](location)\n"
            "    - Book a flight from [London](location) to [Paris.](location)\n",
            2,
        ),
    ],
)
def test_check_correct_entity_annotations(
    text: Text, warnings: int, whitespace_tokenizer: WhitespaceTokenizer
):
    reader = RasaYAMLReader()

    training_data = reader.reads(text)
    whitespace_tokenizer.process_training_data(training_data)

    with pytest.warns(UserWarning) as record:
        EntityExtractorMixin.check_correct_entity_annotations(training_data)

    assert len(record) == warnings
    assert all(
        [excerpt in record[0].message.args[0]]
        for excerpt in ["Misaligned entity annotation in sentence"]
    )
