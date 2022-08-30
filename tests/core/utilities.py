import itertools
import contextlib
import os
import typing
from typing import List, Optional, Text, Any, Dict

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, Event
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY

if typing.TYPE_CHECKING:
    from rasa.shared.core.conversation import Dialogue


def tracker_from_dialogue(dialogue: "Dialogue", domain: Domain) -> DialogueStateTracker:
    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    return tracker


@contextlib.contextmanager
def cwd(path: Text):
    CWD = os.getcwd()

    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(CWD)


@contextlib.contextmanager
def mocked_cmd_input(package, text: Text):
    if isinstance(text, str):
        text = [text]

    text_generator = itertools.cycle(text)
    i = package._get_user_input

    async def mocked_input(*args, **kwargs):
        value = next(text_generator)
        print(f"wrote '{value}' to input")
        return value

    package._get_user_input = mocked_input
    try:
        yield
    finally:
        package._get_user_input = i


def user_uttered(
    text: Text,
    confidence: float = 1.0,
    metadata: Dict[Text, Any] = None,
    timestamp: Optional[float] = None,
) -> UserUttered:
    parse_data = {"intent": {INTENT_NAME_KEY: text, "confidence": confidence}}
    return UserUttered(
        text="Random",
        intent=parse_data["intent"],
        parse_data=parse_data,
        metadata=metadata,
        timestamp=timestamp,
    )


def get_tracker(events: List[Event]) -> DialogueStateTracker:
    return DialogueStateTracker.from_events("sender", events, [], 20)
