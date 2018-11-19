from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.extractors.nested_entity_extractor import NestedEntityExtractor


def test_composite_entities():
    entities = [
        {
            "entity": "meal",
            "value": "rice and chicken"
        },
        {
            "entity": "meal",
            "value": "yam, egg"
        },
        {
            "entity": "meal",
            "value": "noodles"
        },
        {
            "entity": "meal",
            "value": "four pieces"
        },
        {
            "entity": "meal",
            "value": "5 orders"
        },
        {
            "entity": "drink",
            "value": "2 bottles of beer"
        }
    ]
    nested_entities = {
        "lookup_tables": [
            {
                "name": "carbohydrates",
                "elements": [
                    "noodles",
                    "noodle",
                    "rice",
                    "yam"
                ]
            },
            {
                "name": "protein",
                "elements": [
                    "chicken",
                    "eggs",
                    "pork",
                    "egg"
                ]
            }
        ],
        "composite_entities": [
            {
                "composites": [
                    "@protein",
                    "@carbohydrates",
                    "@number"
                ],
                "name": "meal"
            },
            {
                "composites": [
                    "@juice",
                    "@alcohol"
                ],
                "name": "drink"
            },
            {
                "name": "juice",
                "composites": [
                    "@number",
                    "orange",
                    "apple",
                ]
            },
            {
                "name": "alcohol",
                "composites": [
                    "@number",
                    "beer",
                    "spirit",
                ]
            }
        ]
    }
    NestedEntityExtractor(
        nested_entities=nested_entities).split_nested_entities(entities)
    assert len(entities) == 6
    assert entities[0]["value"] == {
        "protein": "chicken",
        "carbohydrates": "rice"
    }
    assert entities[1]["value"] == {
        "protein": "egg",
        "carbohydrates": "yam"
    }
    assert entities[2]["value"] == {
        "carbohydrates": "noodles"
    }
    assert entities[3]["value"] == {
        "number": 4
    }
    assert entities[4]["value"] == {
        "number": 5
    }
    assert entities[5]["value"] == {
        "alcohol": {
            "number": 2
        }
    }
