import os

from decimal import Decimal
from typing import Optional, Text, Union, Any
from pathlib import Path

import pytest

import rasa.core.lock_store
import rasa.utils.io
from rasa.constants import ENV_SANIC_WORKERS
from rasa.core import utils
from rasa.core.lock_store import LockStore, RedisLockStore, InMemoryLockStore
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.domain import Domain
from rasa.utils.endpoints import EndpointConfig
from tests.conftest import write_endpoint_config_to_yaml


class CustomRedisLockStore(RedisLockStore):
    """Test class used to test the behavior of custom lock stores."""


def test_one_hot():
    r = utils.one_hot(4, 6)
    assert (r[[0, 1, 2, 3, 5]] == 0).all()
    assert r[4] == 1


def test_one_hot_out_of_range():
    with pytest.raises(ValueError):
        utils.one_hot(4, 3)


@pytest.mark.parametrize(
    "_input,expected",
    [
        # `int` is not converted
        (-1, -1),
        # `float` is converted
        (2.1, round(Decimal(2.1), 4)),
        # `float` that's too long is rounded
        (1579507733.1107571125030517578125, Decimal("1579507733.110757")),
        # strings are not converted
        (["one", "two"], ["one", "two"]),
        # list of `float`s is converted
        (
            [1.0, -2.1, 3.2],
            [round(Decimal(1.0), 4), round(Decimal(-2.1), 4), round(Decimal(3.2), 4)],
        ),
        # dictionary containing list of `float`s and `float`s is converted
        (
            {"list_with_floats": [4.5, -5.6], "float": 6.7},
            {
                "list_with_floats": [round(Decimal(4.5), 4), round(Decimal(-5.6), 4)],
                "float": round(Decimal(6.7), 4),
            },
        ),
    ],
)
def test_replace_floats_with_decimals(_input: Any, expected: Any):
    assert utils.replace_floats_with_decimals(_input) == expected


@pytest.mark.parametrize(
    "_input,expected",
    [
        # `int` is not converted
        (-1, -1),
        # `float` is converted
        (Decimal(2.1), 2.1),
        # `float` that's too long is rounded to default 9 decimal places
        (Decimal("1579507733.11075711345834582304234"), 1579507733.110757113),
        # strings are not converted
        (["one", "two"], ["one", "two"]),
        # list of `Decimal`s is converted
        ([Decimal(1.0), Decimal(-2.1), Decimal(3.2)], [1.0, -2.1, 3.2]),
        # dictionary containing list of `Decimal`s and `Decimal`s is converted
        (
            {"list_with_floats": [Decimal(4.5), Decimal(-5.6)], "float": Decimal(6.7)},
            {"list_with_floats": [4.5, -5.6], "float": 6.7},
        ),
    ],
)
def test_replace_decimals_with_floats(_input: Any, expected: Any):
    assert utils.replace_decimals_with_floats(_input) == expected


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
        (3, "tests/core/test_utils.CustomRedisLockStore", 3),
        (3, RedisLockStore(), 3),
        (2, InMemoryLockStore(), 1),
        (3, CustomRedisLockStore(), 3),
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
        (EndpointConfig(type="custom_lock_store"), True),
        (None, False),
        (InMemoryLockStore(), False),
        (CustomRedisLockStore(), True),
    ],
)
def test_lock_store_is_multi_worker_compatible(
    lock_store: Union[EndpointConfig, LockStore, None], expected: bool
):
    # noinspection PyProtectedMember
    assert (
        rasa.core.utils._lock_store_is_multi_worker_compatible(lock_store) == expected
    )


def test_read_endpoints_from_path(tmp_path: Path):
    # write valid config to file
    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, {"event_broker": {"type": "pika"}, "tracker_store": {"type": "sql"}}
    )

    # noinspection PyProtectedMember
    available_endpoints = utils.read_endpoints_from_path(endpoints_path)

    # assert event broker and tracker store are valid, others are not
    assert available_endpoints.tracker_store and available_endpoints.event_broker
    assert not all(
        (
            available_endpoints.lock_store,
            available_endpoints.nlg,
            available_endpoints.action,
            available_endpoints.model,
            available_endpoints.nlu,
        )
    )


def test_read_endpoints_from_wrong_path():
    # noinspection PyProtectedMember
    available_endpoints = utils.read_endpoints_from_path("/some/wrong/path")

    # endpoint config is still initialised but does not contain anything
    assert not all(
        (
            available_endpoints.lock_store,
            available_endpoints.nlg,
            available_endpoints.event_broker,
            available_endpoints.tracker_store,
            available_endpoints.action,
            available_endpoints.model,
            available_endpoints.nlu,
        )
    )


def assert_predicted_action(
    prediction: PolicyPrediction,
    domain: Domain,
    expected_action_name: Text,
    confidence: float = 1.0,
    is_end_to_end_prediction: bool = False,
    is_no_user_prediction: bool = False,
) -> None:
    assert prediction.max_confidence == confidence
    index_of_predicted_action = prediction.max_confidence_index
    prediction_action_name = domain.action_names_or_texts[index_of_predicted_action]
    assert prediction_action_name == expected_action_name
    assert prediction.is_end_to_end_prediction == is_end_to_end_prediction
    assert prediction.is_no_user_prediction == is_no_user_prediction
