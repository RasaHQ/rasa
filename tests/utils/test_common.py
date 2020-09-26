import logging

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
