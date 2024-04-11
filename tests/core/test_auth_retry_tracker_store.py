import logging
import sys
from pathlib import Path
from typing import Any, Text, Tuple
from unittest.mock import MagicMock, call

import pytest
from pytest import LogCaptureFixture, MonkeyPatch
from rasa.core.brokers.broker import EventBroker
from rasa.core.tracker_store import AwaitableTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

from rasa.core.auth_retry_tracker_store import (
    DEFAULT_RETRIES,
    AuthRetryTrackerStore,
)
from rasa.core.secrets_manager.secret_manager import EndpointResolver

if sys.version_info[:2] >= (3, 8):
    from unittest.mock import AsyncMock
else:

    class AsyncMock(MagicMock):
        async def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return super().__call__(*args, **kwargs)


@pytest.fixture
def moodbot_domain() -> Domain:
    domain_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "test_domains"
        / "auth_retry_domain.yml"
    )
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

    return tracker_store


@pytest.fixture
def mock_new_tracker_store(moodbot_domain: Domain) -> AsyncMock:
    tracker_store = AsyncMock(
        spec=AwaitableTrackerStore, domain=moodbot_domain, event_broker=None
    )

    tracker_store.save = AsyncMock()
    tracker_store.retrieve = AsyncMock()
    tracker_store.keys = AsyncMock()

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

    domain_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "test_domains"
        / "auth_retry_domain.yml"
    )
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
        "rasa.core.auth_retry_tracker_store.create_tracker_store",
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
