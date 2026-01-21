import pytest

from rasa.shared.constants import DEFAULT_SENDER_ID, DEFAULT_USER_ID
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ACTION_LISTEN_NAME, ActionExecuted, UserUttered
from rasa.shared.core.generator import TrackerWithCachedStates, _subsample_array


@pytest.fixture
def domain() -> Domain:
    return Domain.empty()


def test_subsample_array_read_only():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = _subsample_array(t, 5, can_modify_incoming_array=False)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_subsample_array():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # this will modify the original array and shuffle it
    r = _subsample_array(t, 5)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_cached_tracker_creation_and_copy_with_user_id(domain: Domain):
    """Test TrackerWithCachedStates accepts user_id."""
    tracker = TrackerWithCachedStates(
        DEFAULT_SENDER_ID, [], domain=domain, user_id=DEFAULT_USER_ID
    )
    assert tracker.user_id == DEFAULT_USER_ID
    assert tracker.sender_id == DEFAULT_SENDER_ID

    # Test init_copy preserves user_id
    copy = tracker.init_copy()
    assert copy.user_id == DEFAULT_USER_ID

    # Test full copy preserves user_id and events
    tracker.update(UserUttered("hello"))

    copy = tracker.copy(sender_id="new_sender")

    assert copy.user_id == DEFAULT_USER_ID
    assert copy.sender_id == "new_sender"
    assert len(copy.events) == 1


def test_cached_tracker_creation_and_copy_without_user_id(domain: Domain):
    """Test TrackerWithCachedStates without user_id."""
    tracker = TrackerWithCachedStates(DEFAULT_SENDER_ID, [], domain=domain)
    assert tracker.user_id is None
    assert tracker.sender_id == DEFAULT_SENDER_ID

    # Test init_copy preserves user_id
    copy = tracker.init_copy()
    assert copy.user_id is None

    # Test full copy preserves user_id and events
    tracker.update(UserUttered("hello"))

    copy = tracker.copy(sender_id="new_sender")

    assert copy.user_id is None
    assert copy.sender_id == "new_sender"
    assert len(copy.events) == 1


def test_cached_tracker_from_events_with_user_id(domain: Domain):
    """Test from_events preserves user_id."""
    events = [ActionExecuted(ACTION_LISTEN_NAME)]

    tracker = TrackerWithCachedStates.from_events(
        DEFAULT_SENDER_ID, events, domain=domain, user_id=DEFAULT_USER_ID
    )

    assert tracker.user_id == DEFAULT_USER_ID
    assert len(tracker.events) == 1
    assert tracker.events[0].action_name == ACTION_LISTEN_NAME


def test_cached_tracker_from_events_without_user_id(domain: Domain):
    """Test from_events without user_id."""
    events = [ActionExecuted(ACTION_LISTEN_NAME)]

    tracker = TrackerWithCachedStates.from_events(
        DEFAULT_SENDER_ID, events, domain=domain
    )

    assert tracker.user_id is None
    assert len(tracker.events) == 1
    assert tracker.events[0].action_name == ACTION_LISTEN_NAME
