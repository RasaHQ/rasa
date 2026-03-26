import uuid
from typing import Any, Dict, List, Optional, Tuple

import pytest

from rasa.core.agent import Agent
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.constants import DEFAULT_SENDER_ID
from rasa.shared.core.constants import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message


@pytest.fixture
def test_domain() -> Domain:
    return Domain.load("data/test_domains/default.yml")


async def get_or_create_tracker_store(store: TrackerStore) -> None:
    slot_key = "location"
    slot_val = "Easter Island"

    tracker = await store.get_or_create_tracker(DEFAULT_SENDER_ID)
    ev = SlotSet(slot_key, slot_val)
    tracker.update(ev)
    assert tracker.get_slot(slot_key) == slot_val

    await store.save(tracker)

    again = await store.get_or_create_tracker(DEFAULT_SENDER_ID)
    assert again.get_slot(slot_key) == slot_val


async def create_tracker_with_partially_saved_events(
    tracker_store: TrackerStore,
) -> Tuple[List[Event], DialogueStateTracker]:
    # creates a tracker with two events and saved it to the tracker store
    # following that, it adds three more events that are not saved to the tracker store
    sender_id = uuid.uuid4().hex

    # create tracker with two events and save it
    events = [UserUttered("hello"), BotUttered("what")]
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # add more events to the tracker, do not yet save it
    events = [ActionExecuted(ACTION_LISTEN_NAME), UserUttered("123"), BotUttered("yes")]
    for event in events:
        tracker.update(event)

    return events, tracker


async def _saved_tracker_with_multiple_session_starts(
    tracker_store: TrackerStore, sender_id: str
) -> DialogueStateTracker:
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )

    await tracker_store.save(tracker)
    return await tracker_store.retrieve(sender_id)


async def prepare_token_serialisation(
    tracker_store: TrackerStore, response_selector_agent: Agent, sender_id: str
):
    text = "Good morning"
    tokenizer = WhitespaceTokenizer(WhitespaceTokenizer.get_default_config())
    tokens = tokenizer.tokenize(Message(data={"text": text}), "text")
    indices = [[t.start, t.end] for t in tokens]

    tracker = await tracker_store.get_or_create_tracker(sender_id=sender_id)
    parse_data = await response_selector_agent.parse_message(text)
    event = UserUttered(
        "Good morning",
        parse_data.get("intent"),
        parse_data.get("entities", []),
        parse_data,
    )

    tracker.update(event)
    await tracker_store.save(tracker)

    retrieved_tracker = await tracker_store.retrieve(sender_id=sender_id)
    event = retrieved_tracker.get_last_event_for(event_type=UserUttered)
    event_tokens = event.as_dict().get("parse_data").get("text_tokens")

    assert event_tokens == indices


def sort_key(dialogue_tracker: DialogueStateTracker) -> Tuple:
    """Sorting key for DialogueStateTracker based on first event timestamp and sender_id."""  # noqa: E501
    if dialogue_tracker.events:
        return dialogue_tracker.events[0].timestamp, dialogue_tracker.sender_id
    return 0.0, dialogue_tracker.sender_id


# Helper functions for tracker store user_id tests (shared across all tracker stores)
async def create_tracker_with_user_id(
    tracker_store: TrackerStore,
    sender_id: str,
    user_id: Optional[str],
    events: Optional[List[Event]] = None,
    domain: Optional[Domain] = None,
) -> DialogueStateTracker:
    """Create and save a tracker with optional user_id.

    Works with any TrackerStore implementation (SQL, Redis, MongoDB, DynamoDB, etc.).

    Args:
        tracker_store: The tracker store to save to.
        sender_id: Sender ID for the tracker.
        user_id: Optional user ID to set on the tracker.
        events: Optional list of events.
            Defaults to [SessionStarted(), UserUttered("hello")].
        domain: Optional domain. Uses tracker_store.domain if not provided.

    Returns:
        The created and saved tracker.
    """
    if events is None:
        events = [SessionStarted(), UserUttered("hello")]
    if domain is None:
        domain = tracker_store.domain

    # Some tracker stores (like DynamoDB) need domain passed explicitly
    tracker_kwargs = {"slots": domain.slots, "user_id": user_id}
    if (
        hasattr(tracker_store, "__class__")
        and "Dynamo" in tracker_store.__class__.__name__
    ):
        tracker_kwargs["domain"] = domain

    tracker = DialogueStateTracker.from_events(sender_id, events, **tracker_kwargs)
    await tracker_store.save(tracker)
    return tracker


async def create_multiple_trackers_with_user_id(
    tracker_store: TrackerStore,
    user_id: str,
    count: int,
    start_index: int = 0,
    domain: Optional[Domain] = None,
) -> List[DialogueStateTracker]:
    """Create and save multiple trackers with the same user_id.

    Works with any TrackerStore implementation.

    Args:
        tracker_store: The tracker store to save to.
        user_id: User ID to set on all trackers.
        count: Number of trackers to create.
        start_index: Starting index for sender_id naming (default: 0).
        domain: Optional domain. Uses tracker_store.domain if not provided.

    Returns:
        List of created and saved trackers.
    """
    trackers = []
    for i in range(start_index, start_index + count):
        tracker = await create_tracker_with_user_id(
            tracker_store,
            f"sender{i}",
            user_id,
            [SessionStarted(), UserUttered(f"hello{i}")],
            domain=domain,
        )
        trackers.append(tracker)
    return trackers


def assert_tracker_has_user_id(
    tracker: Dict[str, Any], sender_id: str, user_id: Optional[str]
) -> None:
    """Assert a serialized tracker dict has the correct sender_id and user_id.

    Args:
        tracker: Serialized tracker dict returned by get_trackers_by_user_id.
        sender_id: Expected sender ID.
        user_id: Expected user ID (can be None).
    """
    assert tracker is not None
    assert tracker["sender_id"] == sender_id
    assert tracker["user_id"] == user_id


async def create_tracker_with_explicit_timestamp(
    tracker_store: TrackerStore,
    sender_id: str,
    timestamp: float,
    user_id: Optional[str] = None,
    domain: Optional[Domain] = None,
) -> DialogueStateTracker:
    """Create and save a tracker with explicit timestamp for testing.

    Args:
        tracker_store: The tracker store to save to.
        sender_id: Sender ID for the tracker.
        timestamp: Explicit timestamp to use for events.
        user_id: Optional user ID to set on the tracker.
        domain: Optional domain. Uses tracker_store.domain if not provided.

    Returns:
        The created and saved tracker.
    """
    if domain is None:
        domain = tracker_store.domain

    # Create events with explicit timestamp
    events = [
        SessionStarted(timestamp=timestamp),
        UserUttered("Hello", timestamp=timestamp + 1),
    ]

    # Some tracker stores (like DynamoDB) need domain passed explicitly
    tracker_kwargs = {"slots": domain.slots, "user_id": user_id}
    if (
        hasattr(tracker_store, "__class__")
        and "Dynamo" in tracker_store.__class__.__name__
    ):
        tracker_kwargs["domain"] = domain

    tracker = DialogueStateTracker.from_events(sender_id, events, **tracker_kwargs)
    await tracker_store.save(tracker)
    return tracker


async def create_trackers_with_same_timestamp(
    tracker_store: TrackerStore,
    user_id: str,
    timestamp: float,
    sender_ids: List[str],
    domain: Optional[Domain] = None,
) -> List[DialogueStateTracker]:
    """Create multiple trackers with the same timestamp for sorting tests.

    Args:
        tracker_store: The tracker store to save to.
        user_id: User ID to set on all trackers.
        timestamp: Explicit timestamp to use for all events.
        sender_ids: List of sender IDs to create trackers for.
        domain: Optional domain. Uses tracker_store.domain if not provided.

    Returns:
        List of created and saved trackers.
    """
    trackers = []
    for sender_id in sender_ids:
        tracker = await create_tracker_with_explicit_timestamp(
            tracker_store, sender_id, timestamp, user_id, domain
        )
        trackers.append(tracker)
    return trackers


async def old_tracker_gets_timestamp_on_save(
    tracker_store: TrackerStore,
    sender_id: str = "test_sender",
    domain: Optional[Domain] = None,
) -> None:
    """Helper to test that old tracker without conversation_started_timestamp
    gets it on save.

    Args:
        tracker_store: The tracker store to test with.
        sender_id: Sender ID for the tracker.
        domain: Optional domain. Uses tracker_store.domain if not provided.
    """
    if domain is None:
        domain = tracker_store.domain

    # Create tracker and manually clear timestamp (simulating old tracker)
    tracker_kwargs = {"slots": domain.slots}
    if (
        hasattr(tracker_store, "__class__")
        and "Dynamo" in tracker_store.__class__.__name__
    ):
        tracker_kwargs["domain"] = domain

    tracker = DialogueStateTracker.from_events(
        sender_id,
        [SessionStarted(), UserUttered("Hello")],
        **tracker_kwargs,
    )
    tracker.conversation_started_timestamp = None

    # Save should populate the timestamp
    await tracker_store.save(tracker)

    # Retrieve and verify timestamp was set
    retrieved = await tracker_store.retrieve(sender_id)
    assert retrieved.conversation_started_timestamp is not None
    assert retrieved.conversation_started_timestamp == tracker.events[0].timestamp


async def old_tracker_gets_timestamp_on_update(
    tracker_store: TrackerStore,
    sender_id: str = "test_sender",
    domain: Optional[Domain] = None,
) -> None:
    """Helper to test that old tracker without conversation_started_timestamp
    gets it on update.

    Args:
        tracker_store: The tracker store to test with.
        sender_id: Sender ID for the tracker.
        domain: Optional domain. Uses tracker_store.domain if not provided.
    """
    if domain is None:
        domain = tracker_store.domain

    # Create and save tracker
    tracker_kwargs = {"slots": domain.slots}
    if (
        hasattr(tracker_store, "__class__")
        and "Dynamo" in tracker_store.__class__.__name__
    ):
        tracker_kwargs["domain"] = domain

    tracker = DialogueStateTracker.from_events(
        sender_id,
        [SessionStarted(), UserUttered("Hello")],
        **tracker_kwargs,
    )
    await tracker_store.save(tracker)

    # Manually clear timestamp (simulating old format)
    retrieved = await tracker_store.retrieve(sender_id)
    retrieved.conversation_started_timestamp = None

    # Update should populate the timestamp
    retrieved.update_with_events([UserUttered("Hi")], domain)
    await tracker_store.update(retrieved)

    # Retrieve again and verify timestamp was set
    final = await tracker_store.retrieve(sender_id)
    assert final.conversation_started_timestamp is not None
    assert final.conversation_started_timestamp == final.events[0].timestamp


def assert_all_trackers_have_user_id(trackers: List[Dict], user_id: str) -> None:
    """Assert all trackers in a list have the specified user_id.

    Args:
        trackers: List of trackers to check.
        user_id: Expected user ID.
    """
    for tracker in trackers:
        assert tracker["user_id"] == user_id


async def assert_pagination_results(
    tracker_store: TrackerStore,
    user_id: str,
    saved_trackers: List[DialogueStateTracker],
    skip: Optional[int] = None,
    limit: Optional[int] = None,
    expected_count: Optional[int] = None,
) -> List[DialogueStateTracker]:
    """Helper to test pagination and assert results.

    Args:
        tracker_store: The tracker store to query.
        user_id: User ID to query for.
        saved_trackers: List of saved trackers (for comparison).
        skip: Skip parameter.
        limit: Limit parameter.
        expected_count: Expected number of results. If None, calculated from skip/limit.

    Returns:
        Retrieved trackers.
    """
    trackers = await tracker_store.get_trackers_by_user_id(
        user_id, skip=skip, limit=limit
    )

    # Calculate expected count if not provided
    if expected_count is None:
        sorted_trackers = sorted(saved_trackers, key=sort_key)
        if skip is not None:
            sorted_trackers = sorted_trackers[skip:]
        if limit is not None:
            sorted_trackers = sorted_trackers[:limit]
        expected_count = len(sorted_trackers)

    assert len(trackers) == expected_count
    assert_all_trackers_have_user_id(trackers, user_id)

    return trackers
