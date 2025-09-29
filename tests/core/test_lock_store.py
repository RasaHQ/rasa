import asyncio
import datetime
import sys
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import structlog.testing
from _pytest.monkeypatch import MonkeyPatch
from pydantic import ValidationError

import rasa.core.lock_store
from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.core.constants import (
    DEFAULT_LOCK_LIFETIME,
    ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME,
    IAM_CLOUD_PROVIDER_ENV_VAR_NAME,
)
from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
    AWSElasticacheRedisIAMCredentialsProvider,
)
from rasa.core.lock import Ticket, TicketLock
from rasa.core.lock_store import (
    DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX,
    InMemoryLockStore,
    LockError,
    LockStore,
    RedisLockStore,
    RedisLockStoreConfig,
)
from rasa.core.redis_connection_factory import DeploymentMode
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.utilities import filter_logs


class FakeRedisLockStore(RedisLockStore):
    """Fake `RedisLockStore` using `fakeredis` library."""

    # skipcq: PYL-W0231
    # noinspection PyMissingConstructor
    def __init__(self):
        import fakeredis

        self.red = fakeredis.FakeStrictRedis()

        # added in redis==3.3.0, but not yet in fakeredis
        self.red.connection_pool.connection_class.health_check_interval = 0

        self.key_prefix = DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX


def test_issue_ticket():
    lock = TicketLock("random id 0")

    # no lock issued
    assert lock.last_issued == -1
    assert lock.now_serving == 0

    # no one is waiting
    assert not lock.is_someone_waiting()

    # issue ticket
    ticket = lock.issue_ticket(1)
    assert ticket == 0
    assert lock.last_issued == 0
    assert lock.now_serving == 0

    # someone is waiting
    assert lock.is_someone_waiting()


def test_remove_expired_tickets():
    lock = TicketLock("random id 1")

    # issue one long- and one short-lived ticket
    _ = list(map(lock.issue_ticket, [k for k in [0.01, 10]]))

    # both tickets are there
    assert len(lock.tickets) == 2

    # sleep and only one ticket should be left
    time.sleep(0.02)
    lock.remove_expired_tickets()
    assert len(lock.tickets) == 1


@pytest.mark.parametrize("lock_store", [InMemoryLockStore(), FakeRedisLockStore()])
def test_create_lock_store(lock_store: LockStore):
    conversation_id = "my id 0"

    # create and lock
    lock = lock_store.create_lock(conversation_id)
    lock_store.save_lock(lock)
    lock = lock_store.get_lock(conversation_id)
    assert lock
    assert lock.conversation_id == conversation_id


@pytest.mark.parametrize(
    "redis_response",
    [
        # bytes response (needs conversion)
        b'{"conversation_id": "test_id"}',
        # string response (no conversion needed)
        '{"conversation_id": "test_id"}',
    ],
)
def test_get_lock_handles_bytes_and_string_responses(redis_response: Union[bytes, str]):
    """Test that get_lock properly handles both bytes and string responses."""
    lock_store = RedisLockStore()
    mock_redis = Mock()
    mock_redis.get.return_value = redis_response
    lock_store.red = mock_redis

    result = lock_store.get_lock("test_conversation")

    assert result is not None
    assert result.conversation_id == "test_id"


def test_get_lock_returns_none_for_missing_key():
    """Test that get_lock returns None when Redis returns None."""
    lock_store = RedisLockStore()
    mock_redis = Mock()
    mock_redis.get.return_value = None
    lock_store.red = mock_redis

    result = lock_store.get_lock("nonexistent_conversation")

    assert result is None


def test_raise_connection_exception_redis_lock_store(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        rasa.core.lock_store, "RedisLockStore", Mock(side_effect=ConnectionError())
    )

    with pytest.raises(ConnectionException):
        LockStore.create(
            EndpointConfig(username="username", password="password", type="redis")
        )


@pytest.mark.parametrize("lock_store", [InMemoryLockStore(), FakeRedisLockStore()])
def test_serve_ticket(lock_store: LockStore):
    conversation_id = "my id 1"

    lock = lock_store.create_lock(conversation_id)
    lock_store.save_lock(lock)

    # issue ticket with long lifetime
    ticket_0 = lock_store.issue_ticket(conversation_id, 10)
    assert ticket_0 == 0

    lock = lock_store.get_lock(conversation_id)
    assert lock.last_issued == ticket_0
    assert lock.now_serving == ticket_0
    assert lock.is_someone_waiting()

    # issue another ticket
    ticket_1 = lock_store.issue_ticket(conversation_id, 10)

    # finish serving ticket_0
    lock_store.finish_serving(conversation_id, ticket_0)

    lock = lock_store.get_lock(conversation_id)

    assert lock.last_issued == ticket_1
    assert lock.now_serving == ticket_1
    assert lock.is_someone_waiting()

    # serve second ticket and no one should be waiting
    lock_store.finish_serving(conversation_id, ticket_1)

    lock = lock_store.get_lock(conversation_id)
    assert not lock.is_someone_waiting()


# noinspection PyProtectedMember
@pytest.mark.parametrize("lock_store", [InMemoryLockStore(), FakeRedisLockStore()])
def test_lock_expiration(lock_store: LockStore):
    conversation_id = "my id 2"
    lock = lock_store.create_lock(conversation_id)
    lock_store.save_lock(lock)

    # issue ticket with long lifetime
    ticket = lock.issue_ticket(10)
    assert ticket == 0
    assert not lock._ticket_for_ticket_number(ticket).has_expired()

    # issue ticket with short lifetime
    ticket = lock.issue_ticket(0.00001)
    time.sleep(0.00002)
    assert ticket == 1
    assert lock._ticket_for_ticket_number(ticket) is None

    # newly assigned ticket should get number 1 again
    assert lock.issue_ticket(10) == 1


async def test_multiple_conversation_ids(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'

    conversation_ids = [f"conversation {i}" for i in range(2)]

    # ensure conversations are processed in order
    tasks = [default_agent.handle_text(text, sender_id=_id) for _id in conversation_ids]
    results = await asyncio.gather(*tasks)

    assert results
    processed_ids = [result[0]["recipient_id"] for result in results]
    assert processed_ids == conversation_ids


@pytest.mark.xfail(
    sys.platform == "win32",
    reason="This test sometimes fails on Windows. We want to investigate it further",
)
async def test_message_order(tmp_path: Path, default_agent: Agent):
    start_time = time.time()
    n_messages = 10
    lock_wait = 0.5

    # let's write the incoming order of messages and the order of results to temp files
    results_file = tmp_path / "results_file"
    incoming_order_file = tmp_path / "incoming_order_file"

    # We need to mock `Agent.handle_message()` so we can introduce an
    # artificial holdup (`wait_time_in_seconds`). In the mocked method, we'll
    # record messages as they come and and as they're processed in files so we
    # can check the order later on. We don't need the return value of this method so
    # we'll just return None.
    async def mocked_handle_message(self, message: UserMessage, wait: float) -> None:
        # write incoming message to file
        with open(str(incoming_order_file), "a+") as f_0:
            f_0.write(message.text + "\n")

        async with self.lock_store.lock(
            message.sender_id, wait_time_in_seconds=lock_wait
        ):
            # hold up the message processing after the lock has been acquired
            await asyncio.sleep(wait)

            # write message to file as it's processed
            with open(str(results_file), "a+") as f_1:
                f_1.write(message.text + "\n")

            return None

    # We'll send n_messages from the same sender_id with different blocking times
    # after the lock has been acquired.
    # We have to ensure that the messages are processed in the right order.
    with patch.object(Agent, "handle_message", mocked_handle_message):
        # use decreasing wait times so that every message after the first one
        # does not acquire its lock immediately
        wait_times = np.linspace(0.1, 0.05, n_messages)
        tasks = [
            default_agent.handle_message(
                UserMessage(f"sender {i}", sender_id="some id"), wait=k
            )
            for i, k in enumerate(wait_times)
        ]

        # execute futures
        await asyncio.gather(*(asyncio.ensure_future(t) for t in tasks))

        expected_order = [f"sender {i}" for i in range(len(wait_times))]

        # ensure order of incoming messages is as expected
        with open(str(incoming_order_file)) as f:
            incoming_order = [line for line in f.read().split("\n") if line]
            assert incoming_order == expected_order

        # ensure results are processed in expected order
        with open(str(results_file)) as f:
            results_order = [line for line in f.read().split("\n") if line]
            assert results_order == expected_order

        # Every message after the first one will wait `lock_wait` seconds to acquire its
        # lock (`wait_time_in_seconds` kwarg in `lock_store.lock()`).
        # Let's make sure that this is not blocking and test that total test
        # execution time is less than  the sum of all wait times plus
        # (n_messages - 1) * lock_wait
        time_limit = np.sum(wait_times[1:])
        time_limit += (n_messages - 1) * lock_wait
        assert time.time() - start_time < time_limit


@pytest.mark.xfail(
    sys.platform == "win32",
    reason="This test sometimes fails on Windows. We want to investigate it further",
)
async def test_lock_error(default_agent: Agent):
    lock_lifetime = 0.01
    wait_time_in_seconds = 0.01
    holdup = 0.5

    # Mock message handler again to add a wait time holding up the lock
    # after it's been acquired
    async def mocked_handle_message(self, message: UserMessage) -> None:
        async with self.lock_store.lock(
            message.sender_id,
            wait_time_in_seconds=wait_time_in_seconds,
            lock_lifetime=lock_lifetime,
        ):
            # hold up the message processing after the lock has been acquired
            await asyncio.sleep(holdup)

        return None

    with patch.object(Agent, "handle_message", mocked_handle_message):
        # first message blocks the lock for `holdup`,
        # meaning the second message will not be able to acquire a lock
        tasks = [
            default_agent.handle_message(
                UserMessage(f"sender {i}", sender_id="some id")
            )
            for i in range(2)
        ]

        with pytest.raises(LockError):
            await asyncio.gather(*(asyncio.ensure_future(t) for t in tasks))


async def test_lock_lifetime_environment_variable(monkeypatch: MonkeyPatch):
    import rasa.core.lock_store

    # by default lock lifetime is `DEFAULT_LOCK_LIFETIME`
    assert rasa.core.lock_store._get_lock_lifetime() == DEFAULT_LOCK_LIFETIME

    # set new lock lifetime as environment variable
    new_lock_lifetime = 123
    monkeypatch.setenv("TICKET_LOCK_LIFETIME", str(new_lock_lifetime))

    assert rasa.core.lock_store._get_lock_lifetime() == new_lock_lifetime


@pytest.fixture
def mock_strict_redis(
    monkeypatch: MonkeyPatch,
) -> MagicMock:
    import redis

    _redis_mock = MagicMock(spec=redis.StrictRedis)
    _strict_redis_constructor = MagicMock(return_value=_redis_mock)
    monkeypatch.setattr("redis.StrictRedis", _strict_redis_constructor)
    return _redis_mock


def create_ticket(number: int) -> Ticket:
    """Creates a ticket for testing."""
    return Ticket(
        number=number,
        expires=(
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        ).timestamp(),
    )


def create_ticket_lock(conversation_id: str, tickets: List[Ticket]) -> TicketLock:
    """Creates a TicketLock for testing."""
    return TicketLock(
        conversation_id=conversation_id,
        tickets=deque(tickets),
    )


def create_serialized_lock(tickets: List[Ticket]) -> str:
    """Serialized lock for testing."""
    ticket_lock = create_ticket_lock(
        conversation_id="test_acquire_lock_debug_message",
        tickets=tickets,
    )

    return ticket_lock.dumps()


async def test_redis_lock_store_waiting_lock(
    mock_strict_redis: MagicMock,
):
    """Tests Redis lock store that third lock acquisition will wait for the first two to finish."""  # noqa: E501
    first_ticket = create_ticket(1)
    second_ticket = create_ticket(2)

    already_present_locks = create_serialized_lock([first_ticket, second_ticket])
    lock_with_second_ticket = create_serialized_lock([second_ticket])
    unlock_ticket = create_ticket(3)
    serialized_unlocked_lock = create_serialized_lock([unlock_ticket])

    mock_strict_redis.get.side_effect = [
        # returned on get lock in issue_ticket
        # when lock() receives information that there are already two locks
        # for the same conversation ID, it will issue a ticket for lock with number
        # which is latest_ticket.number + 1
        already_present_locks,
        # returned on first get lock in _acquire_lock,
        # we simulate that first two locks are still being processed
        already_present_locks,
        already_present_locks,  # returned when updating lock
        # we simulate that first lock was processed
        lock_with_second_ticket,  # returned on second get lock _acquire_lock
        lock_with_second_ticket,  # returned when updating lock
        # we simulate that second lock was processed and that there is no one waiting
        serialized_unlocked_lock,  # returned on third get lock in _acquire_lock
        serialized_unlocked_lock,  # returned on finish_serving
        serialized_unlocked_lock,  # returned on delete_lock
    ]

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection",
        return_value=mock_strict_redis,
    ):
        redis_lock = RedisLockStore(RedisLockStoreConfig())
        conversation_id = "test_acquire_lock_debug_message"
        wait_time_in_seconds = 0.01
        with structlog.testing.capture_logs() as caplog:
            async with redis_lock.lock(
                conversation_id, wait_time_in_seconds=wait_time_in_seconds
            ):
                logs = filter_logs(
                    caplog,
                    "lock_store._retrying_lock_acquisition",
                    "debug",
                    [
                        f"because 1 other item(s) for this "
                        f"conversation ID have to be finished "
                        f"processing first. Retrying in "
                        f"{wait_time_in_seconds} seconds ..."
                    ],
                    log_contains_all_message_parts=True,
                )
                assert len(logs) == 1

                logs = filter_logs(
                    caplog,
                    "lock_store._retrying_lock_acquisition",
                    "debug",
                    [
                        f"because 2 other item(s) for this "
                        f"conversation ID have to be finished "
                        f"processing first. Retrying in "
                        f"{wait_time_in_seconds} seconds ..."
                    ],
                    log_contains_all_message_parts=True,
                )
                assert len(logs) == 1


async def test_in_memory_lock_store_waiting_lock():
    """Tests in-memory lock store that third lock acquisition will wait for the first two to finish."""  # noqa: E501
    first_ticket = create_ticket(1)
    second_ticket = create_ticket(2)

    conversation_id = "test_acquire_lock_debug_message"
    already_present_locks = create_ticket_lock(
        conversation_id, [first_ticket, second_ticket]
    )
    lock_with_second_ticket = create_ticket_lock(conversation_id, [second_ticket])
    unlock_ticket = create_ticket(3)
    serialized_unlocked_lock = create_ticket_lock(conversation_id, [unlock_ticket])

    in_memory_lock_store = InMemoryLockStore()
    in_memory_lock_store.get_lock = MagicMock(
        side_effect=[
            # returned on get lock in issue_ticket
            # when lock() receives information
            # that there are already two locks
            # for the same conversation ID,
            # it will issue a ticket for lock with number
            # which is latest_ticket.number + 1
            already_present_locks,
            # returned on first get lock in _acquire_lock,
            # we simulate that first two locks are still being processed
            already_present_locks,
            already_present_locks,  # returned when updating lock
            # we simulate that first lock was processed
            lock_with_second_ticket,  # returned on second get lock _acquire_lock
            lock_with_second_ticket,  # returned when updating lock
            # we simulate that second lock was processed
            # and that there is no one waiting
            serialized_unlocked_lock,  # returned on third get lock in _acquire_lock
            serialized_unlocked_lock,  # returned on finish_serving
            serialized_unlocked_lock,  # returned on delete_lock
        ]
    )
    wait_time_in_seconds = 0.01
    with structlog.testing.capture_logs() as caplog:
        async with in_memory_lock_store.lock(
            conversation_id, wait_time_in_seconds=wait_time_in_seconds
        ):
            logs = filter_logs(
                caplog,
                "lock_store._retrying_lock_acquisition",
                "debug",
                [
                    f"because 1 other item(s) for "
                    f"this conversation ID have to be finished "
                    f"processing first. Retrying "
                    f"in {wait_time_in_seconds} seconds ..."
                ],
                log_contains_all_message_parts=True,
            )
            assert len(logs) == 1

            logs = filter_logs(
                caplog,
                "lock_store._retrying_lock_acquisition",
                "debug",
                [
                    f"because 2 other item(s) for this "
                    f"conversation ID have to be finished "
                    f"processing first. Retrying in "
                    f"{wait_time_in_seconds} seconds ..."
                ],
                log_contains_all_message_parts=True,
            )
            assert len(logs) == 1


async def test_redis_lock_store_timeout(monkeypatch: MonkeyPatch):
    import redis.exceptions

    lock_store = FakeRedisLockStore()
    monkeypatch.setattr(
        lock_store,
        lock_store.get_or_create_lock.__name__,
        Mock(side_effect=redis.exceptions.TimeoutError),
    )

    with pytest.raises(LockError):
        async with lock_store.lock("some sender"):
            pass


async def test_redis_lock_store_with_invalid_prefix(monkeypatch: MonkeyPatch):
    import redis.exceptions

    lock_store = FakeRedisLockStore()

    prefix = "!asdf234 34#"
    lock_store._set_key_prefix(prefix)
    assert lock_store.key_prefix == DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX

    monkeypatch.setattr(
        lock_store,
        lock_store.get_or_create_lock.__name__,
        Mock(side_effect=redis.exceptions.TimeoutError),
    )

    with pytest.raises(LockError):
        async with lock_store.lock("some sender"):
            pass


async def test_redis_lock_store_with_valid_prefix(monkeypatch: MonkeyPatch):
    import redis.exceptions

    lock_store = FakeRedisLockStore()

    prefix = "chatbot42"
    lock_store._set_key_prefix(prefix)
    assert lock_store.key_prefix == prefix + ":" + DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX

    monkeypatch.setattr(
        lock_store,
        lock_store.get_or_create_lock.__name__,
        Mock(side_effect=redis.exceptions.TimeoutError),
    )

    with pytest.raises(LockError):
        async with lock_store.lock("some sender"):
            pass


def test_create_lock_store_from_endpoint_config(endpoints_path: Text):
    store = read_endpoint_config(endpoints_path, endpoint_type="lock_store")

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        lock_store = RedisLockStore(
            config=RedisLockStoreConfig(
                host="localhost",
                port=6379,
                db=0,
                username="username",
                password="password",
                use_ssl=True,
                ssl_keyfile="keyfile.key",
                ssl_certfile="certfile.crt",
                ssl_ca_certs="my-bundle.ca-bundle",
                key_prefix="lock",
            ),
        )

    assert isinstance(lock_store, type(LockStore.create(store)))


@pytest.mark.parametrize(
    "extra_config,expected_mode,expected_endpoints,expected_sentinel_service",
    [
        ({"deployment_mode": "standard"}, DeploymentMode.STANDARD.value, None, None),
        (
            {"deployment_mode": "cluster", "endpoints": ["node1:6379", "node2:6379"]},
            DeploymentMode.CLUSTER.value,
            ["node1:6379", "node2:6379"],
            None,
        ),
        (
            {
                "deployment_mode": "sentinel",
                "endpoints": ["sentinel1:26379"],
                "sentinel_service": "mymaster",
            },
            DeploymentMode.SENTINEL.value,
            ["sentinel1:26379"],
            "mymaster",
        ),
        ({}, DeploymentMode.STANDARD.value, None, None),  # Default case
    ],
)
def test_create_lock_store_deployment_modes(
    extra_config: Dict[str, Any],
    expected_mode: str,
    expected_endpoints: Optional[List[str]],
    expected_sentinel_service: Optional[str],
):
    """Test lock store creation with different deployment modes including default."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        base_config = {"host": "localhost", "port": 6379, "db": 0}
        config = RedisLockStoreConfig(**{**base_config, **extra_config})

        # When
        lock_store = RedisLockStore(config=config)

        # Then
        assert isinstance(lock_store, RedisLockStore)
        assert lock_store.red == mock_redis

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        config = call_args.args[0]

        assert config.deployment_mode == expected_mode
        assert config.endpoints == expected_endpoints
        assert config.sentinel_service == expected_sentinel_service


def test_create_lock_store_default_deployment_mode():
    """Test default lock store creation."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        base_config = {"host": "localhost", "port": 6379, "db": 0}
        config = RedisLockStoreConfig(**{**base_config})

        # When
        lock_store = RedisLockStore(config=config)

        # Then
        assert isinstance(lock_store, RedisLockStore)
        assert lock_store.red == mock_redis

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        config = call_args.args[0]

        assert config.deployment_mode == DeploymentMode.STANDARD.value
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.endpoints is None


def test_redis_lock_store_validation_error():
    """Test that RedisLockStore properly handles configuration validation errors."""

    with pytest.raises(ValidationError) as exc_info:
        config = RedisLockStoreConfig(
            **{
                **{"endpoints": [123, "localhost:6379"]},
            }
        )
        RedisLockStore(config=config)

    assert "validation error for RedisLockStoreConfig" in str(exc_info.value)


@pytest.fixture
def partial_redis_lock_store_config() -> Dict[str, Any]:
    return {
        "type": "redis",
        "port": 6379,
        "db": 0,
        "username": "username",
        "password": "password",
        "use_ssl": True,
        "ssl_keyfile": "keyfile.key",
        "ssl_certfile": "certfile.crt",
        "ssl_ca_certs": "my-bundle.ca-bundle",
        "key_prefix": DEFAULT_REDIS_LOCK_STORE_KEY_PREFIX,
    }


@pytest.fixture
def mock_redis_lock_store(
    monkeypatch: MonkeyPatch,
) -> MagicMock:
    """Mock RedisLockStore to avoid actual Redis connection."""
    mock_store_instance = MagicMock(spec=RedisLockStore)
    mock_constructor = MagicMock(return_value=mock_store_instance)
    monkeypatch.setattr("rasa.core.lock_store.RedisLockStore", mock_constructor)
    return mock_constructor


def test_create_from_endpoint_config_with_host(
    partial_redis_lock_store_config: Dict[str, Any],
    mock_redis_lock_store: MagicMock,
) -> None:
    """Tests that the `host` field is correctly handled."""
    partial_redis_lock_store_config["host"] = "redis://localhost:6379"
    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)
    lock_store = LockStore.create(endpoint_config)

    assert lock_store is mock_redis_lock_store.return_value
    mock_redis_lock_store.assert_called_once_with(
        RedisLockStoreConfig(**endpoint_config.to_dict())
    )


def test_create_from_endpoint_config_url_deprecated(
    partial_redis_lock_store_config: Dict[str, Any],
    mock_redis_lock_store: MagicMock,
) -> None:
    """Tests that the deprecated `url` field is correctly handled."""
    partial_redis_lock_store_config["url"] = "redis://localhost:6379/0"
    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)
    lock_store_config = RedisLockStoreConfig(**endpoint_config.to_dict())

    with pytest.warns() as record:
        lock_store = LockStore.create(endpoint_config)

        assert isinstance(lock_store, RedisLockStore)

        mock_redis_lock_store.assert_called_once_with(lock_store_config)

        future_warnings = [
            warning for warning in record if warning.category == FutureWarning
        ]

        assert len(future_warnings) == 1
        assert (
            "The 'url' property in the redis lock store "
            "configuration is deprecated. Please use 'host' instead."
        ) in str(future_warnings[0].message)


def test_create_from_endpoint_config_host_and_url(
    partial_redis_lock_store_config: Dict[str, Any],
) -> None:
    """Tests that an exception is raised when both `url` and `host` are provided"""
    partial_redis_lock_store_config["url"] = "redis://localhost:6379/0"
    partial_redis_lock_store_config["host"] = "localhost"
    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)

    with pytest.raises(RasaException) as raised_exception:
        LockStore.create(endpoint_config)
        assert (
            "You cannot specify both 'url' and 'host' in the Redis lock store "
            "configuration. Please use only one of them."
        ) in str(raised_exception.value)


def test_redis_lock_store_config_filter_out_unused_arguments(
    partial_redis_lock_store_config: Dict[str, Any],
    mock_redis_lock_store: MagicMock,
) -> None:
    """Tests that unused arguments are filtered out when creating the config."""
    partial_redis_lock_store_config["host"] = "redis://localhost:6379"
    partial_redis_lock_store_config["unused_argument"] = "unused_value"
    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)

    config = RedisLockStoreConfig(**endpoint_config.to_dict())

    result = config.model_dump(
        by_alias=True,
    )

    assert "unused_argument" not in result.keys()


def test_redis_lock_store_config_serialization(
    partial_redis_lock_store_config: Dict[str, Any],
    mock_redis_lock_store: MagicMock,
) -> None:
    """Tests that the RedisLockStoreConfig is serialized correctly."""
    partial_redis_lock_store_config["host"] = "redis://localhost:6379"
    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)

    config = RedisLockStoreConfig(**endpoint_config.to_dict())

    result = config.model_dump(by_alias=True)

    assert str(result["host"]) == partial_redis_lock_store_config["host"]
    assert result["port"] == partial_redis_lock_store_config["port"]
    assert result["db"] == partial_redis_lock_store_config["db"]
    assert result["username"] == partial_redis_lock_store_config["username"]
    assert result["password"] == partial_redis_lock_store_config["password"]
    assert result["ssl"] == partial_redis_lock_store_config["use_ssl"]
    assert result["ssl_keyfile"] == partial_redis_lock_store_config["ssl_keyfile"]
    assert result["ssl_certfile"] == partial_redis_lock_store_config["ssl_certfile"]
    assert result["ssl_ca_certs"] == partial_redis_lock_store_config["ssl_ca_certs"]
    assert result["key_prefix"] == partial_redis_lock_store_config["key_prefix"]


@contextmanager
def not_raises(exception):
    try:
        yield
    except exception:
        raise pytest.fail(f"Raised exception {exception}")


def test_create_from_endpoint_iam_config_no_username_and_password(
    partial_redis_lock_store_config: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    """Username and password not required when using IAM auth."""
    monkeypatch.setenv(ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME, "true")
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.StrictRedis", mock_redis)

    partial_redis_lock_store_config["host"] = "localhost"
    del partial_redis_lock_store_config["username"]
    del partial_redis_lock_store_config["password"]

    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)

    with not_raises(Exception):
        LockStore.create(endpoint_config)

    mock_redis.assert_called_once()
    assert mock_redis.call_args[1].get("credential_provider") is not None
    assert isinstance(
        mock_redis.call_args[1].get("credential_provider"),
        AWSElasticacheRedisIAMCredentialsProvider,
    )


def test_create_from_endpoint_iam_disabled(
    partial_redis_lock_store_config: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests that username and password are required when IAM auth is disabled."""
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.StrictRedis", mock_redis)
    partial_redis_lock_store_config["host"] = "localhost"

    endpoint_config = EndpointConfig(**partial_redis_lock_store_config)

    with not_raises(Exception):
        LockStore.create(endpoint_config)

    mock_redis.assert_called_once()
    assert mock_redis.call_args[1].get("credential_provider") is None
