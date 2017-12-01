# coding=utf-8
import pytest
import logging

from rasa_nlu.evaluate import is_token_within_entity
from rasa_nlu.evaluate import does_token_cross_borders
from rasa_nlu.evaluate import align_entity_predictions
from rasa_nlu.evaluate import determine_intersection
from rasa_nlu.tokenizers import Token

logging.basicConfig(level="DEBUG")

# Chinese Example
# "对面食过敏" -> To be allergic to wheat-based food
CH_wrong_segmentation = [Token(u"对面", 0), Token(u"食", 2), Token(u"过敏", 3)]  # opposite, food, allergy
CH_correct_segmentation = [Token(u"对", 0), Token(u"面食", 1), Token(u"过敏", 3)]  # towards, wheat-based food, allergy
CH_wrong_entity = {
    "start": 0,
    "end": 2,
    "value": u"对面",
    "entity": "direction"

}
CH_correct_entity = {
    "start": 1,
    "end": 3,
    "value": u"面食",
    "entity": "food_type"
}

#EN example
#"Hey Robot, I would like to eat pizza near Alexanderplatz tonight"
EN_indices = [0, 4, 9, 11, 13, 19, 24, 27, 31, 37, 42, 57]
EN_tokens = ["Hey", "Robot", ",", "I", "would", "like", "to", "eat", "pizza", "near", "Alexanderplatz", "tonight"]
EN_tokens = [Token(t, i) for t, i in zip(EN_tokens, EN_indices)]

EN_targets = [
    {
       "start": 31,
        "end": 36,
        "value": "pizza",
        "entity": "food"
    },
    {
        "start": 37,
        "end": 56,
        "value": "near Alexanderplatz",
        "entity": "location"
    },
    {
        "start": 57,
        "end": 64,
        "value": "tonight",
        "entity": "datetime"
    }
]

EN_predicted = [
    {
        "start": 4,
        "end": 9,
        "value": "Robot",
        "entity": "person",
        "extractor": "A"
    },
    {
        "start": 31,
        "end": 36,
        "value": "pizza",
        "entity": "food",
        "extractor": "A"
    },
    {
        "start": 42,
        "end": 56,
        "value": "Alexanderplatz",
        "entity": "location",
        "extractor": "A"
    },
    {
        "start": 42,
        "end": 64,
        "value": "Alexanderplatz tonight",
        "entity": "movie",
        "extractor": "B"
    }
]

def test_token_entity_intersection():
    # included
    assert determine_intersection(CH_correct_segmentation[1], CH_correct_entity) == len(CH_correct_segmentation[1].text)

    # completely outside
    assert determine_intersection(CH_correct_segmentation[2], CH_correct_entity) == 0

    # border crossing
    assert determine_intersection(CH_correct_segmentation[1], CH_wrong_entity) == 1

def test_token_entity_boundaries():
    #smaller and included
    assert is_token_within_entity(CH_wrong_segmentation[1], CH_correct_entity) == True
    assert does_token_cross_borders(CH_wrong_segmentation[1], CH_correct_entity) == False

    # exact match
    assert is_token_within_entity(CH_correct_segmentation[1], CH_correct_entity) == True
    assert does_token_cross_borders(CH_correct_segmentation[1], CH_correct_entity) == False

    # completely outside
    assert is_token_within_entity(CH_correct_segmentation[0], CH_correct_entity) == False
    assert does_token_cross_borders(CH_correct_segmentation[0], CH_correct_entity) == False

    # border crossing
    assert is_token_within_entity(CH_wrong_segmentation[0], CH_correct_entity) == False
    assert does_token_cross_borders(CH_wrong_segmentation[0], CH_correct_entity) == True


def test_evaluate_entities():
    result = align_entity_predictions(EN_targets, EN_predicted, EN_tokens, ["A", "B"])
    assert result == {
        "target_labels": ["O", "O", "O", "O", "O", "O", "O", "O", "food", "location", "location", "datetime"],
        "extractor_labels": {
            "A": ["O", "person", "O", "O", "O", "O", "O", "O", "food", "O", "location", "O"],
            "B": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "movie", "movie"]
        }
    }, "Wrong entity prediction alignment"



