import os
import random

from decimal import Decimal
from typing import Optional, Text, Union

import pytest

import rasa.core.lock_store
import rasa.utils.io
from rasa.constants import ENV_SANIC_WORKERS
from rasa.core import utils
from rasa.core.lock_store import LockStore, RedisLockStore, InMemoryLockStore
from rasa.core.utils import replace_floats_with_decimals
from rasa.utils.endpoints import EndpointConfig


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


def test_float_conversion_to_decimal():
    # Create test objects
    d = {
        "int": -1,
        "float": 2.1,
        "float_round": 1579507733.1107571125030517578125,
        "decimal_round": Decimal("0.92383747394838437473739439744"),
        "list": ["one", "two"],
        "list_of_floats": [1.0, -2.1, 3.2],
        "nested_dict_with_floats": {"list_with_floats": [4.5, -5.6], "float": 6.7},
    }
    d_replaced = replace_floats_with_decimals(d)

    assert isinstance(d_replaced["int"], int)
    assert isinstance(d_replaced["float"], Decimal)
    assert d_replaced["float_round"] == Decimal("1579507733.110757113")
    assert d_replaced["decimal_round"] == Decimal("0.923837474")
    for t in d_replaced["list"]:
        assert isinstance(t, str)
    for f in d_replaced["list_of_floats"]:
        assert isinstance(f, Decimal)
    for f in d_replaced["nested_dict_with_floats"]["list_with_floats"]:
        assert isinstance(f, Decimal)
    assert isinstance(d_replaced["nested_dict_with_floats"]["float"], Decimal)


@pytest.mark.parametrize(
    "env_value,lock_store,expected",
    [
        (1, "redis", 1),
        (4, "redis", 4),
        (None, "redis", 1),
        (0, "redis", 1),
        (-4, "redis", 1),
        ("illegal value", "redis", 1),
        (None, None, 1),
        (None, "in_memory", 1),
        (5, "in_memory", 1),
        (2, None, 1),
        (0, "in_memory", 1),
        (3, RedisLockStore(), 3),
        (2, InMemoryLockStore(), 1),
    ],
)
def test_get_number_of_sanic_workers(
    env_value: Optional[Text],
    lock_store: Union[LockStore, Text, None],
    expected: Optional[int],
):
    # remember pre-test value of SANIC_WORKERS env var
    pre_test_value = os.environ.get(ENV_SANIC_WORKERS)

    # set env var to desired value and make assertion
    if env_value is not None:
        os.environ[ENV_SANIC_WORKERS] = str(env_value)

    # lock_store may be string or LockStore object
    # create EndpointConfig if it's a string, otherwise pass the object
    if isinstance(lock_store, str):
        lock_store = EndpointConfig(type=lock_store)

    assert utils.number_of_sanic_workers(lock_store) == expected

    # reset env var to pre-test value
    os.environ.pop(ENV_SANIC_WORKERS, None)

    if pre_test_value is not None:
        os.environ[ENV_SANIC_WORKERS] = pre_test_value


@pytest.mark.parametrize(
    "lock_store,expected",
    [
        (EndpointConfig(type="redis"), True),
        (RedisLockStore(), True),
        (EndpointConfig(type="in_memory"), False),
        (EndpointConfig(type="random_store"), False),
        (None, False),
        (InMemoryLockStore(), False),
    ],
)
def test_lock_store_is_redis_lock_store(
    lock_store: Union[EndpointConfig, LockStore, None], expected: bool
):
    # noinspection PyProtectedMember
    assert rasa.core.utils._lock_store_is_redis_lock_store(lock_store) == expected


def test_all_subclasses():
    num = random.randint(1, 10)

    class TestClass:
        pass

    classes = [type(f"TestClass{i}", (TestClass,), {}) for i in range(num)]

    assert utils.all_subclasses(TestClass) == classes
