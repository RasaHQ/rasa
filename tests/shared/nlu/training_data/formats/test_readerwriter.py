
import pytest
from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataWriter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from typing import Text, List, Dict, Any


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "message_text, expected_text, entities",
    [
        (
            "I like chocolate",
            "I like chocolate",
            [],
        ),
        (
            "I like chocolate",
            "I like [chocolate](food)",
            [
                {"entity": "food", "value": "chocolate", "start": 7, "end": 16},
            ],
        ),
        (
            "I like chocolate",
            "I like [chocolate]{\"entity\": \"food\", \"value\": \"desert\"}",
            [
                {"entity": "food", "value": "desert", "start": 7, "end": 16},
            ],
        ),
        (
            f"{INTENT_MESSAGE_PREFIX}I like chocolate",
            f"{INTENT_MESSAGE_PREFIX}I like chocolate",
            [
                {"entity": "food", "value": "desert", "start": 7, "end": 16},
            ],
        ),
        (
            "I like chocolate",
            "I like [chocolate][{\"entity\": \"food\"}, {\"entity\": \"desert\"}]",
            [
                {"entity": "food", "value": "chocolate", "start": 7, "end": 16},
                {"entity": "desert", "value": "chocolate", "start": 7, "end": 16},
            ],
        ),
    ]
)
def test_generate_message(
    message_text: Text,
    expected_text: Text,
    entities: List[Dict[Text, Any]],
):
    message = Message.build(
        message_text,
        "dummy_intent",
        entities=entities
    )
    message_text = TrainingDataWriter.generate_message(message)

    assert message_text == expected_text


@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "message_text, entities",
    [
        (
            "I like chocolate cake",
            [
                {"entity": "food", "value": "chocolate", "start": 7, "end": 16},
                {"entity": "desert", "value": "chocolate cake", "start": 7, "end": 21},
            ],
        ),
    ]
)
def test_generate_message_raises_on_overlapping_but_not_identical_spans(
    message_text: Text,
    entities: List[Dict[Text, Any]],
):
    message = Message.build(
        message_text,
        "dummy_intent",
        entities=entities
    )
    with pytest.raises(ValueError):
        TrainingDataWriter.generate_message(message)
