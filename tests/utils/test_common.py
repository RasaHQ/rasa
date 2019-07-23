import rasa.utils.common as common_utils


def test_sort_dicts_by_keys():
    test_data = [{"Z": 1}, {"A": 10}]

    expected = [{"A": 10}, {"Z": 1}]
    actual = common_utils.sort_list_of_dicts_by_first_key(test_data)

    assert actual == expected
