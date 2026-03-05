import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Text, Tuple
from unittest.mock import MagicMock, call

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from rasa.core.brokers.broker import EventBroker
from rasa.core.secrets_manager.secret_manager import EndpointResolver
from rasa.core.tracker_stores.auth_retry_tracker_store import (
    DEFAULT_RETRIES,
    AuthRetryTrackerStore,
)
from rasa.core.tracker_stores.tracker_store import AwaitableTrackerStore, TrackerStore
from rasa.shared.constants import DEFAULT_USER_ID
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

if sys.version_info[:2] >= (3, 8):
    from unittest.mock import AsyncMock
else:

    class AsyncMock(MagicMock):
        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return super().__call__(*args, **kwargs)


@pytest.fixture
def moodbot_domain() -> Domain:
    domain_path = Path("data/test_domains/auth_retry_domain.yml")
    return Domain.load(domain_path)


@pytest.fixture
def credentials() -> Tuple[Text, Text]:
    return "myusername", "mypassword"


@pytest.fixture
def sender_id() -> Text:
    return "unit_test_auth_retry"


@pytest.fixture
def tracker(sender_id: Text) -> DialogueStateTracker:
    events = [
        UserUttered("hello", {"name": "greet"}),
        ActionExecuted("utter_greet"),
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id=sender_id, evts=events, slots=[]
    )
    return tracker


@pytest.fixture
def mock_tracker_store(moodbot_domain: Domain) -> AsyncMock:
    tracker_store = AsyncMock(
        spec=AwaitableTrackerStore, domain=moodbot_domain, event_broker=None
    )

    tracker_store.save = AsyncMock(side_effect=Exception)
    tracker_store.retrieve = AsyncMock(side_effect=Exception)
    tracker_store.keys = AsyncMock(side_effect=Exception)
    tracker_store.update = AsyncMock(side_effect=Exception)
    tracker_store.delete = AsyncMock(side_effect=Exception)
    tracker_store.retrieve_full_tracker = AsyncMock(side_effect=Exception)

    return tracker_store


@pytest.fixture
def mock_new_tracker_store(moodbot_domain: Domain) -> AsyncMock:
    tracker_store = AsyncMock(
        spec=AwaitableTrackerStore, domain=moodbot_domain, event_broker=None
    )

    tracker_store.save = AsyncMock()
    tracker_store.retrieve = AsyncMock()
    tracker_store.keys = AsyncMock()
    tracker_store.update = AsyncMock()
    tracker_store.delete = AsyncMock()
    tracker_store.retrieve_full_tracker = AsyncMock()

    return tracker_store


@pytest.fixture
def mock_auth_retry_tracker_store_recreate_tracker_store(
    monkeypatch: MonkeyPatch,
) -> MagicMock:
    _recreate_tracker_store = MagicMock()
    monkeypatch.setattr(
        AuthRetryTrackerStore,
        "recreate_tracker_store",
        _recreate_tracker_store,
    )

    return _recreate_tracker_store


@pytest.fixture(autouse=True)
def reset_auth_retry_tracker_store_class_attribute() -> None:
    AuthRetryTrackerStore.endpoint_config = None


def test_auth_retry_tracker_store_init(
    mock_tracker_store: AsyncMock,
    moodbot_domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_recreate_tracker_store = MagicMock()
    mock_recreate_tracker_store.return_value = mock_tracker_store
    endpoint_config = EndpointConfig()
    monkeypatch.setattr(
        AuthRetryTrackerStore,
        "recreate_tracker_store",
        mock_recreate_tracker_store,
    )

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=endpoint_config, domain=moodbot_domain, retries=5
    )
    assert auth_retry_tracker_store._tracker_store == mock_tracker_store
    assert auth_retry_tracker_store.retries == 5
    assert auth_retry_tracker_store.domain == mock_tracker_store.domain
    assert auth_retry_tracker_store.endpoint_config == endpoint_config


def test_auth_retry_tracker_store_init_invalid_retries(
    mock_tracker_store: AsyncMock,
    caplog: LogCaptureFixture,
    moodbot_domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_recreate_tracker_store = MagicMock()
    mock_recreate_tracker_store.return_value = mock_tracker_store
    endpoint_config = EndpointConfig()
    monkeypatch.setattr(
        AuthRetryTrackerStore,
        "recreate_tracker_store",
        mock_recreate_tracker_store,
    )

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=endpoint_config, domain=moodbot_domain, retries=-1
    )

    log_msg = (
        f"Invalid number of retries: -1. "
        f"Using default number of retries: {DEFAULT_RETRIES}"
    )
    assert log_msg in caplog.text
    assert auth_retry_tracker_store._tracker_store == mock_tracker_store
    assert auth_retry_tracker_store.retries == DEFAULT_RETRIES
    assert auth_retry_tracker_store.domain == mock_tracker_store.domain
    assert auth_retry_tracker_store.endpoint_config == endpoint_config


def test_auth_retry_tracker_store_domain_property(
    moodbot_domain: Domain,
) -> None:
    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(), domain=moodbot_domain, retries=2
    )
    assert auth_retry_tracker_store.domain == moodbot_domain
    assert auth_retry_tracker_store._tracker_store.domain == moodbot_domain


def test_auth_retry_tracker_store_domain_setter() -> None:
    # we need to test updating the domain on the tracker store since that
    # is what Rasa OSS does when it hot-reloads a model when run in
    # server mode. it will unpack the model and set the domain of a new
    # model on the already loaded tracker store - we got to make sure it
    # populates to the wrapped tracker
    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(), domain=Domain.empty(), retries=2
    )
    assert auth_retry_tracker_store.domain.is_empty()

    domain_path = Path("data/test_domains/auth_retry_domain.yml")
    new_domain = Domain.load(domain_path)
    auth_retry_tracker_store.domain = new_domain

    assert not auth_retry_tracker_store.domain.is_empty()
    assert auth_retry_tracker_store.domain == new_domain
    assert auth_retry_tracker_store._tracker_store.domain == new_domain


@pytest.fixture
def mock_create_tracker_store(
    monkeypatch: MonkeyPatch, mock_tracker_store: MagicMock
) -> MagicMock:
    _mock_create_tracker_store = MagicMock()
    _mock_create_tracker_store.return_value = mock_tracker_store

    return _mock_create_tracker_store


@pytest.fixture
def set_mock_create_tracker_store(
    mock_create_tracker_store: MagicMock, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rasa.core.tracker_stores.auth_retry_tracker_store.create_tracker_store",
        mock_create_tracker_store,
    )


def test_auth_retry_tracker_store_recreate_tracker_store(
    moodbot_domain: Domain,
    credentials: Tuple[Text, Text],
    monkeypatch: MonkeyPatch,
    mock_tracker_store: AsyncMock,
    mock_create_tracker_store: MagicMock,
    set_mock_create_tracker_store: None,
) -> None:
    updated_config = EndpointConfig(url="new_url")
    mock_update_config = MagicMock()
    mock_update_config.return_value = updated_config
    monkeypatch.setattr(
        EndpointResolver,
        "update_config",
        mock_update_config,
    )

    original_config = EndpointConfig(url="old_url")
    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=original_config, domain=moodbot_domain
    )
    tracker_store = auth_retry_tracker_store.recreate_tracker_store(moodbot_domain)

    mock_create_tracker_store.assert_called_with(updated_config, moodbot_domain, None)

    assert mock_tracker_store == tracker_store
    mock_update_config.assert_called_with(original_config)


async def test_auth_retry_tracker_store_save(
    mock_tracker_store: AsyncMock,
    moodbot_domain: Domain,
    sender_id: Text,
    tracker: DialogueStateTracker,
    caplog: LogCaptureFixture,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
) -> None:
    mock_tracker_store.save = AsyncMock(side_effect=None)
    mock_auth_retry_tracker_store_recreate_tracker_store.return_value = (
        mock_tracker_store
    )
    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(), domain=moodbot_domain, retries=1
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.save(tracker)

    assert caplog.text == ""


async def test_auth_retry_tracker_store_save_successful_with_exception(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_new_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: Text,
    tracker: DialogueStateTracker,
    caplog: LogCaptureFixture,
) -> None:
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.save(tracker)

    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
        ]
    )

    mock_tracker_store.save.assert_called_once_with(tracker)

    log_msg = f"Failed to save tracker for {sender_id}. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_save_unsuccessful_after_max_retries(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: Text,
    tracker: DialogueStateTracker,
    caplog: LogCaptureFixture,
) -> None:
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        await auth_retry_tracker_store.save(tracker)

    mock_tracker_store.save.assert_has_calls(
        [
            call(tracker),
            call(tracker),
        ]
    )
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
        ]
    )

    log_msg = f"Failed to save tracker for {sender_id} after {retries} retries."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_retrieve(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    sender_id: Text,
    caplog: LogCaptureFixture,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
) -> None:
    mock_tracker_store.retrieve = AsyncMock(side_effect=None)
    mock_auth_retry_tracker_store_recreate_tracker_store.return_value = (
        mock_tracker_store
    )

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(), domain=moodbot_domain, retries=1
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.retrieve(sender_id)

    assert caplog.text == ""


async def test_auth_retry_tracker_store_retrieve_successful_with_exception(
    mock_tracker_store: AsyncMock,
    moodbot_domain: Domain,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    mock_new_tracker_store: AsyncMock,
    sender_id: Text,
    caplog: LogCaptureFixture,
) -> None:
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.retrieve(sender_id=sender_id)

    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
        ]
    )

    mock_tracker_store.retrieve.assert_called_once_with(sender_id)

    log_msg = f"Failed to retrieve tracker for {sender_id}. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_retrieve_unsuccessful_after_max_retries(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: Text,
    caplog: LogCaptureFixture,
) -> None:
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        await auth_retry_tracker_store.retrieve(sender_id=sender_id)

    mock_tracker_store.retrieve.assert_has_calls(
        [
            call(sender_id),
            call(sender_id),
        ]
    )
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
        ]
    )

    log_msg = f"Failed to retrieve tracker for {sender_id} after {retries} retries."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_keys(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    sender_id: Text,
    tracker: DialogueStateTracker,
    caplog: LogCaptureFixture,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
) -> None:
    mock_tracker_store.keys = AsyncMock(side_effect=None)
    mock_auth_retry_tracker_store_recreate_tracker_store.return_value = (
        mock_tracker_store
    )

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(), domain=moodbot_domain, retries=1
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.keys()

    assert caplog.text == ""


async def test_auth_retry_tracker_store_keys_successful_with_exception(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    mock_new_tracker_store: AsyncMock,
    sender_id: Text,
    caplog: LogCaptureFixture,
) -> None:
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]

    mock_new_tracker_store.keys.return_value = [sender_id]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.keys()

    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
        ]
    )

    mock_tracker_store.keys.assert_called_once()

    log_msg = "Failed to retrieve keys. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_keys_unsuccessful_after_max_retries(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: Text,
    caplog: LogCaptureFixture,
) -> None:
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        await auth_retry_tracker_store.keys()

    assert mock_tracker_store.keys.call_count == 2

    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
        ]
    )

    log_msg = f"Failed to retrieve keys after {retries} retries."
    assert log_msg in caplog.text


async def test_wrapper_tracker_stores_delete(monkeypatch: MonkeyPatch) -> None:
    mocked_inner_tracker_store = MagicMock(spec=TrackerStore)
    monkeypatch.setattr(
        "rasa.core.tracker_stores.auth_retry_tracker_store.AuthRetryTrackerStore.recreate_tracker_store",
        lambda *args, **kwargs: mocked_inner_tracker_store,
    )
    tracker_store = AuthRetryTrackerStore(mocked_inner_tracker_store, EndpointConfig())

    mocked_inner_tracker_store.delete = AsyncMock()
    sender_id = uuid.uuid4().hex
    await tracker_store.delete(sender_id)
    mocked_inner_tracker_store.delete.assert_called_once_with(sender_id)


async def test_wrapper_tracker_stores_update(monkeypatch: MonkeyPatch) -> None:
    mocked_inner_tracker_store = MagicMock(spec=TrackerStore)
    monkeypatch.setattr(
        "rasa.core.tracker_stores.auth_retry_tracker_store.AuthRetryTrackerStore.recreate_tracker_store",
        lambda *args, **kwargs: mocked_inner_tracker_store,
    )
    tracker_store = AuthRetryTrackerStore(mocked_inner_tracker_store, EndpointConfig())

    mocked_inner_tracker_store.update = AsyncMock()
    tracker = DialogueStateTracker.from_events(
        sender_id="test_sender",
        evts=[UserUttered("test message")],
    )
    await tracker_store.update(tracker)
    mocked_inner_tracker_store.update.assert_called_once_with(
        tracker, apply_deletion_only=True
    )


async def test_auth_retry_tracker_store_update_successful_with_exception(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    mock_new_tracker_store: AsyncMock,
    sender_id: str,
    caplog: LogCaptureFixture,
) -> None:
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        tracker = DialogueStateTracker.from_events(
            sender_id=sender_id,
            evts=[UserUttered("test message")],
        )
        await auth_retry_tracker_store.update(tracker)

    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
        ]
    )

    mock_tracker_store.update.assert_called_once_with(tracker, apply_deletion_only=True)

    log_msg = f"Failed to replace tracker for {sender_id}. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_update_unsuccessful_after_max_retries(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: str,
    caplog: LogCaptureFixture,
) -> None:
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        tracker = DialogueStateTracker.from_events(
            sender_id=sender_id,
            evts=[UserUttered("test message")],
        )
        await auth_retry_tracker_store.update(tracker)

    assert mock_tracker_store.update.call_count == 2

    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
        ]
    )

    log_msg = (
        f"Failed to replace tracker for {tracker.sender_id} "
        f"after {retries} retries."
    )
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_delete_successful_with_exception(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    mock_new_tracker_store: AsyncMock,
    sender_id: str,
    caplog: LogCaptureFixture,
) -> None:
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.delete(sender_id)

    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
        ]
    )

    mock_tracker_store.delete.assert_called_once_with(sender_id)

    log_msg = f"Failed to delete tracker for {sender_id}. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_delete_unsuccessful_after_max_retries(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: str,
    caplog: LogCaptureFixture,
) -> None:
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        await auth_retry_tracker_store.delete(sender_id)

    assert mock_tracker_store.delete.call_count == 2

    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
        ]
    )

    log_msg = f"Failed to delete tracker for {sender_id} " f"after {retries} retries."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_retrieve_full_tracker_successful_with_exception(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    mock_new_tracker_store: AsyncMock,
    sender_id: str,
    caplog: LogCaptureFixture,
) -> None:
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        await auth_retry_tracker_store.retrieve_full_tracker(sender_id)

    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
            call(
                auth_retry_tracker_store.domain,
                auth_retry_tracker_store.event_broker,
            ),
        ]
    )

    mock_tracker_store.retrieve_full_tracker.assert_called_once_with(sender_id)

    log_msg = f"Failed to retrieve full tracker for {sender_id}. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_retrieve_full_tracker_unsuccessful_after_max_retries(  # noqa: E501
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    sender_id: str,
    caplog: LogCaptureFixture,
) -> None:
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        await auth_retry_tracker_store.retrieve_full_tracker(sender_id)

    assert mock_tracker_store.retrieve_full_tracker.call_count == 2

    mock_auth_retry_tracker_store_recreate_tracker_store.assert_has_calls(
        [
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
            call(
                auth_retry_tracker_store.domain, auth_retry_tracker_store.event_broker
            ),
        ]
    )

    log_msg = (
        f"Failed to retrieve full tracker for {sender_id} " f"after {retries} retries."
    )
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_get_trackers_by_user_id(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    caplog: LogCaptureFixture,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
) -> None:
    """Test get_trackers_by_user_id delegates to wrapped tracker store."""
    user_id = DEFAULT_USER_ID
    expected_trackers = [
        DialogueStateTracker.from_events(
            "sender1", [UserUttered("hello")], user_id=user_id
        ),
        DialogueStateTracker.from_events(
            "sender2", [UserUttered("hi")], user_id=user_id
        ),
    ]
    mock_tracker_store.get_trackers_by_user_id = AsyncMock(
        return_value=expected_trackers
    )
    mock_auth_retry_tracker_store_recreate_tracker_store.return_value = (
        mock_tracker_store
    )

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(), domain=moodbot_domain, retries=1
    )

    with caplog.at_level(logging.WARNING):
        result = await auth_retry_tracker_store.get_trackers_by_user_id(user_id)

    assert result == expected_trackers
    mock_tracker_store.get_trackers_by_user_id.assert_called_once_with(
        user_id, limit=None, skip=None
    )
    assert caplog.text == ""


async def test_auth_retry_tracker_store_get_trackers_by_user_id_successful_with_exc(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    mock_new_tracker_store: AsyncMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test get_trackers_by_user_id retries on exception."""
    user_id = DEFAULT_USER_ID
    expected_trackers = [
        DialogueStateTracker.from_events(
            "sender1", [UserUttered("hello")], user_id=user_id
        ),
    ]
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_new_tracker_store,
    ]
    mock_tracker_store.get_trackers_by_user_id.side_effect = Exception("DB error")
    mock_new_tracker_store.get_trackers_by_user_id = AsyncMock(
        return_value=expected_trackers
    )

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=1,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.WARNING):
        result = await auth_retry_tracker_store.get_trackers_by_user_id(user_id)

    assert result == expected_trackers
    assert auth_retry_tracker_store._tracker_store == mock_new_tracker_store
    mock_tracker_store.get_trackers_by_user_id.assert_called_once_with(
        user_id, limit=None, skip=None
    )
    mock_new_tracker_store.get_trackers_by_user_id.assert_called_once_with(
        user_id, limit=None, skip=None
    )

    log_msg = f"Failed to retrieve trackers for user_id {user_id}. Retrying..."
    assert log_msg in caplog.text


async def test_auth_retry_tracker_store_get_trackers_by_user_id_raise_after_max_retries(
    moodbot_domain: Domain,
    mock_tracker_store: AsyncMock,
    mock_auth_retry_tracker_store_recreate_tracker_store: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    """Test get_trackers_by_user_id returns empty list after max retries."""
    user_id = "user_123"
    retries = 1
    mock_auth_retry_tracker_store_recreate_tracker_store.side_effect = [
        mock_tracker_store,
        mock_tracker_store,
        mock_tracker_store,
    ]

    mock_tracker_store.get_trackers_by_user_id.side_effect = [Exception("DB error")]

    auth_retry_tracker_store = AuthRetryTrackerStore(
        endpoint_config=EndpointConfig(),
        domain=moodbot_domain,
        retries=retries,
        event_broker=EventBroker(),
    )

    with caplog.at_level(logging.DEBUG):
        result = await auth_retry_tracker_store.get_trackers_by_user_id(user_id)

    assert result == []
    assert mock_tracker_store.get_trackers_by_user_id.call_count == 2

    log_msg = (
        f"Failed to retrieve trackers for user_id {user_id} "
        f"after {retries} retries."
    )
    assert log_msg in caplog.text
