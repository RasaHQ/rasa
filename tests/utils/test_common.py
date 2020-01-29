import pytest

from rasa.utils.common import raise_warning, sort_list_of_dicts_by_first_key


def test_sort_dicts_by_keys():
    test_data = [{"Z": 1}, {"A": 10}]

    expected = [{"A": 10}, {"Z": 1}]
    actual = sort_list_of_dicts_by_first_key(test_data)

    assert actual == expected


def test_raise_warning():
    with pytest.warns(UserWarning) as record:
        raise_warning("My warning.")

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."


def test_raise_deprecation():
    with pytest.warns(DeprecationWarning) as record:
        raise_warning("My warning.", DeprecationWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."
    assert isinstance(record[0].message, DeprecationWarning)
