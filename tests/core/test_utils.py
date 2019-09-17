import asyncio

import pytest

import rasa.utils.io
from rasa.core import utils


@pytest.fixture(scope="session")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


def test_is_int():
    assert utils.is_int(1)
    assert utils.is_int(1.0)
    assert not utils.is_int(None)
    assert not utils.is_int(1.2)
    assert not utils.is_int("test")


def test_subsample_array_read_only():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = utils.subsample_array(t, 5, can_modify_incoming_array=False)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_subsample_array():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # this will modify the original array and shuffle it
    r = utils.subsample_array(t, 5)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_on_hot():
    r = utils.one_hot(4, 6)
    assert (r[[0, 1, 2, 3, 5]] == 0).all()
    assert r[4] == 1


def test_on_hot_out_of_range():
    with pytest.raises(ValueError):
        utils.one_hot(4, 3)


def test_cap_length():
    assert utils.cap_length("mystring", 6) == "mys..."


def test_cap_length_without_ellipsis():
    assert utils.cap_length("mystring", 3, append_ellipsis=False) == "mys"


def test_cap_length_with_short_string():
    assert utils.cap_length("my", 3) == "my"


def test_read_lines():
    lines = utils.read_lines(
        "data/test_stories/stories.md", max_line_limit=2, line_pattern=r"\*.*"
    )

    lines = list(lines)

    assert len(lines) == 2


def test_pad_lists_to_size():
    list_x = [1, 2, 3]
    list_y = ["a", "b"]
    list_z = [None, None, None]

    assert utils.pad_lists_to_size(list_x, list_y) == (list_x, ["a", "b", None])
    assert utils.pad_lists_to_size(list_y, list_x, "c") == (["a", "b", "c"], list_x)
    assert utils.pad_lists_to_size(list_z, list_x) == (list_z, list_x)


def test_convert_bytes_to_string():
    # byte string will be decoded
    byte_string = b"\xcf\x84o\xcf\x81\xce\xbdo\xcf\x82"
    decoded_string = "τoρνoς"
    assert utils.convert_bytes_to_string(byte_string) == decoded_string

    # string remains string
    assert utils.convert_bytes_to_string(decoded_string) == decoded_string
