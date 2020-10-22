import itertools

import contextlib
import typing
from typing import List, Optional, Text, Any, Dict

import jsonpickle
import os

import rasa.shared.utils.io
import rasa.utils.io
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered, Event
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS

if typing.TYPE_CHECKING:
    from rasa.shared.core.conversation import Dialogue


def tracker_from_dialogue_file(
    filename: Text, domain: Optional[Domain] = None
) -> DialogueStateTracker:
    dialogue = read_dialogue_file(filename)

    if not domain:
        domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)

    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    return tracker


def read_dialogue_file(filename: Text) -> "Dialogue":
    return jsonpickle.loads(rasa.shared.utils.io.read_file(filename))


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
    i = package.get_user_input

    def mocked_input(*args, **kwargs):
        value = next(text_generator)
        print(f"wrote '{value}' to input")
        return value

    package.get_user_input = mocked_input
    try:
        yield
    finally:
        package.get_user_input = i


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
