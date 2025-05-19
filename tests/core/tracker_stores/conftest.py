import uuid
from typing import List, Tuple

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
