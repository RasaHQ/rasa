from typing import Collection, List, Text

import pytest

import rasa.shared.utils.common


def test_all_subclasses():
    class TestClass:
        pass

    subclasses = [type(f"Subclass{i}", (TestClass,), {}) for i in range(10)]
    sub_subclasses = [
        type(f"Sub-subclass_{subclass.__name__}", (subclass,), {})
        for subclass in subclasses
    ]

    expected = subclasses + sub_subclasses
    assert rasa.shared.utils.common.all_subclasses(TestClass) == expected


def test_sort_dicts_by_keys():
    test_data = [{"Z": 1}, {"A": 10}]

    expected = [{"A": 10}, {"Z": 1}]
    actual = rasa.shared.utils.common.sort_list_of_dicts_by_first_key(test_data)

    assert actual == expected


@pytest.mark.parametrize(
    "collection, possible_outputs",
    [
        (["a", "b", "c"], ["a, b and c"]),
        (["a", "b"], ["a and b"]),
        (["a"], ["a"]),
        (
            {"a", "b", "c"},
            [
                "a, b and c",
                "a, c and b",
                "b, a and c",
                "b, c and a",
                "c, a and b",
                "c, b and a",
            ],
        ),
        ({"a", "b"}, ["a and b", "b and a"]),
        ({"a"}, ["a"]),
        ({}, [""]),
    ],
)
def test_transform_collection_to_sentence(
    collection: Collection, possible_outputs: List[Text]
):
    actual = rasa.shared.utils.common.transform_collection_to_sentence(collection)
    assert actual in possible_outputs
