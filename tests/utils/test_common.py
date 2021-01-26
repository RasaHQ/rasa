import logging
from typing import Any

import rasa.utils.common
from rasa.utils.common import RepeatedLogFilter


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


async def test_call_maybe_coroutine_with_async() -> Any:
    expected = 5

    async def my_function():
        return expected

    actual = await rasa.utils.common.call_potential_coroutine(my_function())

    assert actual == expected


async def test_call_maybe_coroutine_with_sync() -> Any:
    expected = 5

    def my_function():
        return expected

    actual = await rasa.utils.common.call_potential_coroutine(my_function())

    assert actual == expected
