from typing import Text
import pytest
import numpy as np
import scipy.sparse
import rasa.shared.nlu.training_data.util


@pytest.mark.parametrize(
    "s, has_escaped_char",
    [
        ("Hey,\nmy name is Christof", True),
        ("Howdy!", False),
        ("A\tB", True),
        ("Hey,\rmy name is Thomas", True),
        ("Hey, my name is Thomas", False),
        ("Hey,\nI\ncan\nwrite\nmany\nlines.", True),
    ],
)
def test_has_string_escape_chars(s: Text, has_escaped_char: bool):
    assert (
        rasa.shared.nlu.training_data.util.has_string_escape_chars(s)
        == has_escaped_char
    )


def test_sparse_matrix_to_string():
    m = np.zeros((9, 9))
    m[0, 4] = 5.0
    m[3, 3] = 6.0
    m_sparse = scipy.sparse.csr_matrix(m)
    expected_result = "  (0, 4)\t5.0\n  (3, 3)\t6.0"
    result = rasa.shared.nlu.training_data.util.sparse_matrix_to_string(m_sparse)
    assert result == expected_result
