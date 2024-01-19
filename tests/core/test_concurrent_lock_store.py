import logging
from pathlib import Path

import rasa.utils.endpoints
from _pytest.logging import LogCaptureFixture
from rasa.core.lock_store import LockStore
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION

from rasa.core.concurrent_lock_store import (
    DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX,
    ConcurrentRedisLockStore,
)


def test_create_concurrent_redis_lock_store(tmp_path: Path) -> None:
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    lock_store = LockStore.create(endpoint_config)

    assert isinstance(lock_store, ConcurrentRedisLockStore)


def test_create_concurrent_redis_lock_store_valid_custom_key_prefix(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    endpoints_file = tmp_path / "endpoints.yml"
    custom_prefix = "testPrefix"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
            key_prefix: {custom_prefix}
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    with caplog.at_level(logging.DEBUG):
        lock_store = LockStore.create(endpoint_config)

    assert f"Setting non-default redis key prefix: '{custom_prefix}'." in caplog.text
    assert (
        lock_store.key_prefix
        == custom_prefix + ":" + DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX
    )


def test_create_concurrent_redis_lock_store_invalid_custom_key_preifx(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    endpoints_file = tmp_path / "endpoints.yml"
    invalid_prefix = "test_prefix"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
            key_prefix: {invalid_prefix}
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    with caplog.at_level(logging.WARNING):
        lock_store = LockStore.create(endpoint_config)

    assert (
        f"Omitting provided non-alphanumeric redis key prefix: '{invalid_prefix}'."
        in caplog.text
    )
    assert lock_store.key_prefix == DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX
