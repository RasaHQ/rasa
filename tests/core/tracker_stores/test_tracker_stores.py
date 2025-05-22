# file deepcode ignore NoHardcodedCredentials/test: Secrets are all just examples for tests. # noqa: E501

import uuid
from typing import List, Optional, Text, Tuple, Type
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from pytest import FixtureRequest, MonkeyPatch
from structlog.testing import capture_logs

import rasa.core.tracker_stores.sql_tracker_store
import rasa.core.tracker_stores.tracker_store
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.tracker_stores.sql_tracker_store import SQLTrackerStore
from rasa.core.tracker_stores.tracker_store import (
    AwaitableTrackerStore,
    FailSafeTrackerStore,
    InMemoryTrackerStore,
    TrackerStore,
)
from rasa.plugin import plugin_manager
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    DialogueStackUpdated,
    Event,
    Restarted,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker, TrackerEventDiffEngine
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from tests.core.tracker_stores.conftest import (
    get_or_create_tracker_store,
    prepare_token_serialisation,
)
from tests.utilities import filter_logs


def test_get_or_create(test_domain: Domain):
    get_or_create_tracker_store(InMemoryTrackerStore(test_domain))


async def test_restart_after_retrieval_from_tracker_store(domain: Domain):
    store = InMemoryTrackerStore(domain)
    tr = await store.get_or_create_tracker("myuser")
    synth = [ActionExecuted("action_listen") for _ in range(4)]

    for e in synth:
        tr.update(e)

    tr.update(Restarted())
    latest_restart = tr.idx_after_latest_restart()

    await store.save(tr)
    tr2 = await store.retrieve("myuser")
    latest_restart_after_loading = tr2.idx_after_latest_restart()
    assert latest_restart == latest_restart_after_loading


async def test_tracker_store_remembers_max_history(domain: Domain):
    store = InMemoryTrackerStore(domain)
    tr = await store.get_or_create_tracker("myuser", max_event_history=42)
    tr.update(Restarted())

    await store.save(tr)
    tr2 = await store.retrieve("myuser")
    assert tr._max_event_history == tr2._max_event_history == 42


def test_tracker_store_endpoint_config_loading(endpoints_path: Text):
    cfg = read_endpoint_config(endpoints_path, "tracker_store")

    assert cfg == EndpointConfig.from_dict(
        {
            "type": "redis",
            "url": "localhost",
            "port": 6379,
            "db": 0,
            "username": "username",
            "password": "password",
            "timeout": 30000,
            "use_ssl": True,
            "ssl_keyfile": "keyfile.key",
            "ssl_certfile": "certfile.crt",
            "ssl_ca_certs": "my-bundle.ca-bundle",
        }
    )


class NonAsyncTrackerStore(TrackerStore):
    def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        pass

    def save(self, tracker: DialogueStateTracker) -> None:
        pass


def test_tracker_store_from_invalid_module(domain: Domain, monkeypatch: MonkeyPatch):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "a.module.which.cannot.be.found"

    def mock_create_tracker_store(*args, **kwargs):
        return None

    monkeypatch.setattr(
        plugin_manager().hook, "create_tracker_store", mock_create_tracker_store
    )

    with pytest.warns(UserWarning):
        tracker_store = TrackerStore.create(store_config, domain)

    assert isinstance(tracker_store, InMemoryTrackerStore)


def test_tracker_store_from_invalid_string(domain: Domain, monkeypatch: MonkeyPatch):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = "any string"

    def mock_create_tracker_store(*args, **kwargs):
        return None

    monkeypatch.setattr(
        plugin_manager().hook, "create_tracker_store", mock_create_tracker_store
    )

    with pytest.warns(UserWarning):
        tracker_store = TrackerStore.create(store_config, domain)

    assert isinstance(tracker_store, InMemoryTrackerStore)


async def _tracker_store_and_tracker_with_slot_set(
    test_domain: Domain,
) -> Tuple[InMemoryTrackerStore, DialogueStateTracker]:
    # returns an InMemoryTrackerStore containing a tracker with a slot set

    slot_key = "cuisine"
    slot_val = "French"

    store = InMemoryTrackerStore(test_domain)
    tracker = await store.get_or_create_tracker(DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)

    return store, tracker


async def test_tracker_serialisation(test_domain: Domain):
    store, tracker = await _tracker_store_and_tracker_with_slot_set(test_domain)
    serialised = store.serialise_tracker(tracker)

    assert tracker == store.deserialise_tracker(DEFAULT_SENDER_ID, serialised)


async def test_fail_safe_tracker_store_if_no_errors():
    mocked_tracker_store = Mock()

    tracker_store = FailSafeTrackerStore(mocked_tracker_store, None)

    # test save
    mocked_tracker_store.save = AsyncMock()
    await tracker_store.save(None)
    mocked_tracker_store.save.assert_called_once()

    # test retrieve
    expected = [1]
    mocked_tracker_store.retrieve = AsyncMock(return_value=expected)
    sender_id = "10"
    assert await tracker_store.retrieve(sender_id) == expected
    mocked_tracker_store.retrieve.assert_called_once_with(sender_id)

    # test keys
    expected = ["sender 1", "sender 2"]
    mocked_tracker_store.keys = AsyncMock(return_value=expected)
    assert await tracker_store.keys() == expected
    mocked_tracker_store.keys.assert_called_once()


async def test_fail_safe_tracker_store_with_save_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.save = Mock(side_effect=Exception())

    fallback_tracker_store = Mock()
    fallback_tracker_store.save = AsyncMock()

    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(
        mocked_tracker_store, on_error_callback, fallback_tracker_store
    )
    await tracker_store.save(None)

    fallback_tracker_store.save.assert_called_once()
    on_error_callback.assert_called_once()


async def test_fail_safe_tracker_store_with_keys_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.keys = Mock(side_effect=Exception())

    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(mocked_tracker_store, on_error_callback)
    assert await tracker_store.keys() == []
    on_error_callback.assert_called_once()


async def test_fail_safe_tracker_store_with_retrieve_error():
    mocked_tracker_store = Mock()
    mocked_tracker_store.retrieve = Mock(side_effect=Exception())

    fallback_tracker_store = Mock()
    on_error_callback = Mock()

    tracker_store = FailSafeTrackerStore(
        mocked_tracker_store, on_error_callback, fallback_tracker_store
    )

    assert await tracker_store.retrieve("sender_id") is None
    on_error_callback.assert_called_once()


def test_set_fail_safe_tracker_store_domain(domain: Domain):
    tracker_store = InMemoryTrackerStore(domain)
    fallback_tracker_store = InMemoryTrackerStore(None)
    failsafe_store = FailSafeTrackerStore(tracker_store, None, fallback_tracker_store)

    failsafe_store.domain = domain
    assert failsafe_store.domain is domain
    assert tracker_store.domain is failsafe_store.domain
    assert fallback_tracker_store.domain is failsafe_store.domain


async def test_in_memory_store_retrieve_with_events_from_previous_sessions() -> None:
    tracker_store = InMemoryTrackerStore(Domain.empty())

    conversation_id = uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )
    await tracker_store.save(tracker)

    actual = await tracker_store.retrieve_full_tracker(conversation_id)

    assert len(actual.events) == len(tracker.events)


async def test_in_memory_tracker_store_counts_conversations() -> None:
    tracker_store = InMemoryTrackerStore(Domain.empty())

    # Create two trackers
    tracker1 = DialogueStateTracker.from_events("1", [SessionStarted(timestamp=1)])
    tracker2 = DialogueStateTracker.from_events("2", [SessionStarted(timestamp=3)])
    await tracker_store.save(tracker1)
    await tracker_store.save(tracker2)

    # Assert that the tracker store counts the conversations correctly
    assert await tracker_store.count_conversations() == 2
    assert await tracker_store.count_conversations(after_timestamp=2) == 1
    assert await tracker_store.count_conversations(after_timestamp=4) == 0

    # Create another tracker
    tracker3 = DialogueStateTracker.from_events("3", [SessionStarted(timestamp=5)])
    await tracker_store.save(tracker3)

    # Assert that the tracker store counts the conversations correctly
    assert await tracker_store.count_conversations() == 3
    assert await tracker_store.count_conversations(after_timestamp=4) == 1


def test_in_memory_tracker_store_with_token_serialisation(
    domain: Domain, default_agent: Agent
):
    tracker_store = InMemoryTrackerStore(domain)
    prepare_token_serialisation(tracker_store, default_agent, "inmemory")


def test_create_non_async_tracker_store(domain: Domain, monkeypatch: MonkeyPatch):
    endpoint_config = EndpointConfig(
        type="tests.core.tracker_stores.test_tracker_stores.NonAsyncTrackerStore"
    )

    def mock_create_tracker_store(*args, **kwargs):
        return None

    monkeypatch.setattr(
        plugin_manager().hook, "create_tracker_store", mock_create_tracker_store
    )

    with pytest.warns(FutureWarning):
        tracker_store = TrackerStore.create(endpoint_config)
    assert isinstance(tracker_store, AwaitableTrackerStore)
    assert isinstance(tracker_store._tracker_store, NonAsyncTrackerStore)


def test_create_awaitable_tracker_store_with_endpoint_config():
    endpoint_config = EndpointConfig(
        type="tests.core.tracker_stores.test_tracker_stores.NonAsyncTrackerStore"
    )
    tracker_store = AwaitableTrackerStore.create(endpoint_config)

    assert isinstance(tracker_store, AwaitableTrackerStore)
    assert isinstance(tracker_store._tracker_store, NonAsyncTrackerStore)


def test_create_tracker_store_from_endpoints_file_in_memory_tracker_store(
    domain: Domain,
) -> None:
    endpoint_config = read_endpoint_config("", "tracker_store")
    tracker_store = rasa.core.tracker_stores.tracker_store.create_tracker_store(
        endpoint_config, domain
    )

    assert (
        rasa.core.tracker_stores.tracker_store.check_if_tracker_store_async(
            tracker_store
        )
        is True
    )
    assert isinstance(tracker_store, InMemoryTrackerStore)


async def test_fail_safe_tracker_store_retrieve_full_tracker(
    domain: Domain, tracker_with_restarted_event: DialogueStateTracker
) -> None:
    primary_tracker_store = SQLTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    tracker_store = FailSafeTrackerStore(primary_tracker_store)
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_fail_safe_tracker_store_retrieve_full_tracker_with_exception() -> None:
    primary_tracker_store = MagicMock()
    primary_tracker_store.domain = Domain.empty()
    primary_tracker_store.event_broker = None

    exception = Exception("Something went wrong")
    primary_tracker_store.retrieve_full_tracker = AsyncMock(side_effect=exception)

    tracker_store = FailSafeTrackerStore(primary_tracker_store)
    with capture_logs() as caplog:
        await tracker_store.retrieve_full_tracker("some_id")
        logs = filter_logs(
            caplog,
            event="fail_safe_tracker_store.tracker_store_retrieve_error",
            log_level="error",
            log_message_parts=[
                "Error happened when trying to retrieve conversation tracker"
            ],
            exec_info=exception,
        )
        assert len(logs) == 1


async def test_in_memory_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = InMemoryTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_in_memory_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = InMemoryTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert list(tracker.events) == events_after_restart


async def test_tracker_event_diff_engine_event_difference() -> None:
    start_session_sequence = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    events: List[Event] = start_session_sequence + [UserUttered("hello")]
    prior_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )
    new_events = [BotUttered("Hey! How can I help you?")]
    events += new_events

    new_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )

    event_diff = TrackerEventDiffEngine.event_difference(prior_tracker, new_tracker)

    assert new_events == event_diff


async def test_tracker_store_retrieve_stack_events():
    tracker_store = InMemoryTrackerStore(domain=Domain.empty())
    tracker = DialogueStateTracker.from_events(
        "test_patterns",
        [
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "PWF4YX9P", "flow_id": "list_contacts", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0/previous_flow_name", "value": "list your contacts"}, {"op": "replace", "path": "/0/type", "value": "pattern_completed"}]'  # noqa: E501
            ),
        ],
    )

    await tracker_store.save(tracker)

    retrieved_tracker = await tracker_store.retrieve_full_tracker(tracker.sender_id)

    assert retrieved_tracker == tracker


async def test_in_memory_tracker_store_delete() -> None:
    # Given
    tracker_store = InMemoryTrackerStore(domain=Domain.empty())
    sender_id = uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(sender_id, [SessionStarted()])
    await tracker_store.save(tracker)

    # When
    await tracker_store.delete(sender_id)

    # Then
    retrieved_tracker = await tracker_store.retrieve(sender_id)
    assert retrieved_tracker is None


@pytest.mark.parametrize(
    "tracker_store_type",
    [
        FailSafeTrackerStore,
        AwaitableTrackerStore,
    ],
)
async def test_wrapper_tracker_stores_delete(
    tracker_store_type: Type[TrackerStore],
) -> None:
    mocked_inner_tracker_store = Mock()
    tracker_store = tracker_store_type(mocked_inner_tracker_store)

    mocked_inner_tracker_store.delete = AsyncMock()
    sender_id = uuid.uuid4().hex
    await tracker_store.delete(sender_id)
    mocked_inner_tracker_store.delete.assert_called_once_with(sender_id)


@pytest.fixture(
    params=[
        "data/test_endpoints/event_brokers/kafka_pii_endpoint.yml",
        "data/test_endpoints/event_brokers/pika_with_pii_endpoint.yml",
    ]
)
async def mock_event_broker_no_pii(
    request: FixtureRequest, monkeypatch: MonkeyPatch
) -> EventBroker:
    """Fixture to create an event broker with stream_pii set to False."""
    if "pika" in request.param:
        # Mock RabbitMQ connection
        monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock())
    cfg = read_endpoint_config(request.param, "event_broker")
    return await EventBroker.create(cfg)


async def test_tracker_store_stream_events_no_pii(
    mock_event_broker_no_pii: EventBroker,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests that the tracker store streams events without PII."""
    tracker_store = InMemoryTrackerStore(Domain.empty(), mock_event_broker_no_pii)
    mock_stream_new_events = AsyncMock()
    monkeypatch.setattr(tracker_store, "_stream_new_events", mock_stream_new_events)
    tracker = DialogueStateTracker.from_events(
        "test_no_pii", [ActionExecuted("action_listen")]
    )

    with capture_logs() as caplog:
        await tracker_store.stream_events(tracker)
        logs = filter_logs(
            caplog,
            event="tracker_store.stream_events.no_streaming",
            log_level="debug",
            log_message_parts=[
                "Un-anonymized events will not be published to the event broker."
            ],
        )
        assert len(logs) == 1

    mock_stream_new_events.assert_not_called()


@pytest.fixture(
    params=[
        "data/test_endpoints/event_brokers/kafka_plaintext_endpoint.yml",
        "data/test_endpoints/event_brokers/pika_endpoint.yml",
    ]
)
async def mock_event_broker(
    request: FixtureRequest, monkeypatch: MonkeyPatch
) -> EventBroker:
    """Fixture to create an event broker with stream_pii set to True."""
    if "pika" in request.param:
        # Mock RabbitMQ connection
        monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock())
    cfg = read_endpoint_config(request.param, "event_broker")
    return await EventBroker.create(cfg)


async def test_tracker_store_stream_events_with_pii(
    mock_event_broker: EventBroker,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests that the tracker store streams events with PII."""
    # Given
    tracker_store = InMemoryTrackerStore(Domain.empty(), mock_event_broker)
    mock_stream_new_events = AsyncMock()
    monkeypatch.setattr(tracker_store, "_stream_new_events", mock_stream_new_events)
    tracker = DialogueStateTracker.from_events(
        "test_with_pii", [ActionExecuted("action_listen")]
    )

    # When
    await tracker_store.stream_events(tracker)
    # Then
    mock_stream_new_events.assert_called_once()
