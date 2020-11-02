import asyncio
from typing import Collection, List, Text
from unittest.mock import Mock

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
        async def f(self, arg1: bool, arg2: bool):
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
