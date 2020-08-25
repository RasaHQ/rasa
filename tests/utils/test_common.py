import logging
from typing import Collection, List, Text

import pytest

from rasa.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from rasa.utils.common import (
    raise_warning,
    raise_deprecation_warning,
    sort_list_of_dicts_by_first_key,
    transform_collection_to_sentence,
    RepeatedLogFilter,
)


def test_sort_dicts_by_keys():
    test_data = [{"Z": 1}, {"A": 10}]

    expected = [{"A": 10}, {"Z": 1}]
    actual = sort_list_of_dicts_by_first_key(test_data)

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
    actual = transform_collection_to_sentence(collection)
    assert actual in possible_outputs


def test_raise_user_warning():
    with pytest.warns(UserWarning) as record:
        raise_warning("My warning.")

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."


def test_raise_future_warning():
    with pytest.warns(FutureWarning) as record:
        raise_warning("My future warning.", FutureWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My future warning."


def test_raise_deprecation():
    with pytest.warns(DeprecationWarning) as record:
        raise_warning("My warning.", DeprecationWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."
    assert isinstance(record[0].message, DeprecationWarning)


def test_raise_deprecation_warning():
    with pytest.warns(FutureWarning) as record:
        raise_deprecation_warning(
            "This feature is deprecated.", warn_until_version="3.0.0"
        )

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "This feature is deprecated. (will be removed in 3.0.0)"
    )


def test_raise_deprecation_warning_version_already_in_message():
    with pytest.warns(FutureWarning) as record:
        raise_deprecation_warning(
            "This feature is deprecated and will be removed in 3.0.0!",
            warn_until_version="3.0.0",
        )

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "This feature is deprecated and will be removed in 3.0.0!"
    )


def test_raise_deprecation_warning_default():
    with pytest.warns(FutureWarning) as record:
        raise_deprecation_warning("This feature is deprecated.")

    assert len(record) == 1
    assert record[0].message.args[0] == (
        f"This feature is deprecated. "
        f"(will be removed in {NEXT_MAJOR_VERSION_FOR_DEPRECATIONS})"
    )


def test_repeated_log_filter():
    log_filter = RepeatedLogFilter()
    record1 = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Super msg: %s", ("yes",), None
    )
    record1_same = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Super msg: %s", ("yes",), None
    )
    record2_other_args = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Super msg: %s", ("no",), None
    )
    record3_other = logging.LogRecord(
        "rasa", logging.INFO, "/some/path.py", 42, "Other msg", (), None
    )
    assert log_filter.filter(record1) is True
    assert log_filter.filter(record1_same) is False  # same log
    assert log_filter.filter(record2_other_args) is True
    assert log_filter.filter(record3_other) is True
    assert log_filter.filter(record1) is True  # same as before, but not repeated
