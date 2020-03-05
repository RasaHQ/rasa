import pytest

import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.nlu.constants import BILOU_ENTITIES, ENTITIES
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data import TrainingData, Message


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("B-person", "person"),
        ("I-location", "location"),
        ("location", "location"),
        ("U-company", "company"),
        ("L-company", "company"),
    ],
)
def test_entity_name_from_tag(tag, expected):
    actual = bilou_utils.entity_name_from_tag(tag)

    assert actual == expected


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("B-person", "B"),
        ("I-location", "I"),
        ("location", None),
        ("U-company", "U"),
        ("L-company", "L"),
        ("O-company", None),
    ],
)
def test_bilou_from_tag(tag, expected):
    actual = bilou_utils.bilou_prefix_from_tag(tag)

    assert actual == expected


def test_tags_to_ids():
    message = Message("Germany is part of the European Union")
    message.set(
        BILOU_ENTITIES,
        ["U-location", "O", "O", "O", "O", "B-organisation", "L-organisation"],
    )

    tag_id_dict = {"O": 0, "U-location": 1, "B-organisation": 2, "L-organisation": 3}

    tags = bilou_utils.tags_to_ids(message, tag_id_dict)

    assert tags == [1, 0, 0, 0, 0, 2, 3]


def test_remove_bilou_prefixes():
    actual = bilou_utils.remove_bilou_prefixes(
        ["U-location", "O", "O", "O", "O", "B-organisation", "L-organisation"]
    )

    assert actual == ["location", "O", "O", "O", "O", "organisation", "organisation"]


def test_build_tag_id_dict():
    message_1 = Message("Germany is part of the European Union")
    message_1.set(
        BILOU_ENTITIES,
        ["U-location", "O", "O", "O", "O", "B-organisation", "L-organisation"],
    )

    message_2 = Message("Berlin is the capital of Germany")
    message_2.set(BILOU_ENTITIES, ["U-location", "O", "O", "O", "O", "U-location"])

    training_data = TrainingData([message_1, message_2])

    tag_id_dict = bilou_utils.build_tag_id_dict(training_data)

    assert tag_id_dict == {
        "O": 0,
        "B-location": 1,
        "I-location": 2,
        "U-location": 3,
        "L-location": 4,
        "B-organisation": 5,
        "I-organisation": 6,
        "U-organisation": 7,
        "L-organisation": 8,
    }


def test_apply_bilou_schema():
    tokenizer = WhitespaceTokenizer()

    message_1 = Message("Germany is part of the European Union")
    message_1.set(
        ENTITIES,
        [
            {"start": 0, "end": 7, "value": "Germany", "entity": "location"},
            {
                "start": 23,
                "end": 37,
                "value": "European Union",
                "entity": "organisation",
            },
        ],
    )

    message_2 = Message("Berlin is the capital of Germany")
    message_2.set(
        ENTITIES,
        [
            {"start": 0, "end": 6, "value": "Berlin", "entity": "location"},
            {"start": 25, "end": 32, "value": "Germany", "entity": "location"},
        ],
    )

    training_data = TrainingData([message_1, message_2])

    tokenizer.train(training_data)

    bilou_utils.apply_bilou_schema(training_data)

    assert message_1.get(BILOU_ENTITIES) == [
        "U-location",
        "O",
        "O",
        "O",
        "O",
        "B-organisation",
        "L-organisation",
        "O",
    ]
    assert message_2.get(BILOU_ENTITIES) == [
        "U-location",
        "O",
        "O",
        "O",
        "O",
        "U-location",
        "O",
    ]
