import asyncio
from typing import Collection, List, Optional, Text
from unittest.mock import Mock

import pytest

import rasa.shared.core.domain
import rasa.shared.utils.common
from rasa.shared.exceptions import RasaException


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


async def test_cached_method_with_sync_method():
    expected = 5
    mock = Mock(return_value=expected)

    class Test:
        @rasa.shared.utils.common.cached_method
        def f(self):
            return mock()

    test_instance = Test()
    assert test_instance.f() == expected
    assert test_instance.f() == expected

    mock.assert_called_once()


async def test_cached_method_with_async_method():
    expected = 5
    mock = Mock(return_value=expected)

    class Test:
        @rasa.shared.utils.common.cached_method
        async def f(self):
            await asyncio.sleep(0)
            return mock()

    test_instance = Test()
    assert await test_instance.f() == expected
    assert await test_instance.f() == expected

    mock.assert_called_once()


async def test_cached_method_with_different_arguments():
    expected = 5
    mock = Mock(return_value=expected)

    class Test:
        @rasa.shared.utils.common.cached_method
        async def f(self, _: bool, arg2: bool):
            return mock()

    test_instance = Test()

    # Caching works
    assert await test_instance.f(True, arg2=True) == expected
    assert await test_instance.f(True, arg2=True) == expected

    assert mock.call_count == 1

    # Different arg results in cache miss
    assert await test_instance.f(False, arg2=True) == expected
    assert await test_instance.f(False, arg2=True) == expected

    assert mock.call_count == 2

    # Different kwarg results in cache miss
    assert await test_instance.f(True, arg2=False) == expected
    assert await test_instance.f(True, arg2=False) == expected

    assert mock.call_count == 3


def test_cached_method_with_function():
    with pytest.raises(AssertionError):

        @rasa.shared.utils.common.cached_method
        def my_function():
            pass

        my_function()


async def test_cached_method_with_async_function():
    with pytest.raises(AssertionError):

        @rasa.shared.utils.common.cached_method
        async def my_function():
            await asyncio.sleep(0)

        await my_function()


@pytest.mark.parametrize(
    "module_path, lookup_path, outcome",
    [
        ("rasa.shared.core.domain.Domain", None, "Domain"),
        # lookup_path
        ("Event", "rasa.shared.core.events", "Event"),
    ],
)
def test_class_from_module_path(
    module_path: Text, lookup_path: Optional[Text], outcome: Text
):
    klass = rasa.shared.utils.common.class_from_module_path(module_path, lookup_path)
    assert isinstance(klass, object)
    assert klass.__name__ == outcome


@pytest.mark.parametrize(
    "module_path, lookup_path",
    [
        ("rasa.shared.core.domain.FunkyDomain", None),
        ("FunkyDomain", None),
        ("FunkyDomain", "rasa.shared.core.domain"),
    ],
)
def test_class_from_module_path_not_found(
    module_path: Text, lookup_path: Optional[Text]
):
    with pytest.raises(ImportError):
        rasa.shared.utils.common.class_from_module_path(module_path, lookup_path)


def test_class_from_module_path_fails():
    module_path = "rasa.shared.core.domain.logger"
    with pytest.raises(RasaException):
        rasa.shared.utils.common.class_from_module_path(module_path)


def test_extract_duplicates():
    list_one = ["greet", {"inform": {"use_entities": []}}, "start_form", "goodbye"]
    list_two = ["goodbye", {"inform": {"use_entities": ["destination"]}}]

    expected = ["goodbye", "inform"]
    result = rasa.shared.utils.common.extract_duplicates(list_one, list_two)

    assert result == expected


def test_extract_duplicates_with_unique_lists():
    list_one = ["greet", {"inform": {"use_entities": []}}, "start_form", "goodbye"]
    list_two = ["bot_challenge", {"mood_sad": {"ignore_entities": []}}]

    result = rasa.shared.utils.common.extract_duplicates(list_one, list_two)
    assert result == []


def test_clean_duplicates():
    duplicates = {"intents": ["goodbye", "inform"], "entities": []}
    expected = {"intents": ["goodbye", "inform"]}
    result = rasa.shared.utils.common.clean_duplicates(duplicates)
    assert result == expected


def test_merge_lists():
    list_one = ["greet", "start_form", "goodbye"]
    list_two = ["goodbye", "bot_challenge", "greet"]
    expected = ["bot_challenge", "goodbye", "greet", "start_form"]
    result = rasa.shared.utils.common.merge_lists(list_one, list_two)

    assert result == expected


@pytest.mark.parametrize("override_existing_values", [False, True])
def test_merge_dicts(override_existing_values):
    dict_1 = {"intents": ["greet", "goodbye"], "entities": ["name"]}
    dict_2 = {
        "responses": {"utter_greet": [{"text": "Hi"}]},
        "intents": ["bot_challenge"],
    }

    if override_existing_values:
        expected = {
            "entities": ["name"],
            "intents": ["bot_challenge"],
            "responses": {"utter_greet": [{"text": "Hi"}]},
        }
    else:
        expected = {
            "entities": ["name"],
            "intents": ["greet", "goodbye"],
            "responses": {"utter_greet": [{"text": "Hi"}]},
        }

    result = rasa.shared.utils.common.merge_dicts(
        dict_1, dict_2, override_existing_values
    )

    assert result == expected


@pytest.mark.parametrize("override_existing_values", [False, True])
def test_merge_lists_of_dicts(override_existing_values):
    list_one = ["greet", {"inform": {"use_entities": []}}, "start_form", "goodbye"]
    list_two = ["goodbye", {"inform": {"use_entities": ["destination"]}}]

    if override_existing_values:
        expected = [
            "greet",
            {"inform": {"use_entities": ["destination"]}},
            "start_form",
            "goodbye",
        ]
    else:
        expected = ["goodbye", {"inform": {"use_entities": []}}, "greet", "start_form"]

    result = rasa.shared.utils.common.merge_lists_of_dicts(
        list_one, list_two, override_existing_values
    )

    assert result == expected
