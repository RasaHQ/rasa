"""gen_conversations.py

This script generates random conversations, and stores them into an event
broker. It is similar to the `rasa export` command, but instead of reading
tracker events from a tracker store, it generates them by selecting random
text messages. To select an event broker to use, point this script at your
`endpoints.yml` with an `event_broker` section.

Usage help:

$ python3 scripts/gen_conversations.py --help

"""

import uuid
import time
import random
from itertools import count as counter
from typing import List, Text
import argparse

from tqdm import tqdm

import rasa.cli.utils as cli_utils
from rasa.core.exporter import Exporter
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.trackers import DialogueStateTracker
from rasa.core.brokers.broker import EventBroker
from rasa.core.utils import AvailableEndpoints
from rasa.core.events import (
    ActionExecuted,
    SessionStarted,
    UserUttered,
    BotUttered,
    Event,
)


MESSAGE_TEXTS = [
    "Yes.",
    "Yes?",
    "Mmmhh... maybe.",
    "Yes, quite.",
    "No?",
    "No.",
    "Don't think so.",
    "Don't think so, no.",
    "Why do you ask that?",
    "Say that again please.",
    "What?",
    "What???",
    "Really?",
    "So yeah",
    "Could you repeat that?",
    "Could you repeat that please?",
    "That's amazing.",
    "I agree.",
    "I don't agree.",
    "That makes sense.",
    "Why not?",
    "Why?",
    "I was actually referring to another thing.",
    "Yes, that.",
    "Did you hear about that?",
    "Did you get that?",
    "No, not that.",
    "Well, yes.",
    "Well, no.",
    "Do you think so?",
    "That's not very nice.",
    "That's nice.",
    "Hello!",
    "Hi",
    "Hey there",
    "I want to leave",
    "I want end this conversation",
    "I don't really like you, to be honest",
    "Tell me about you.",
    "Are you liking this conversation?",
    "Haha",
    "Sounds good!",
    "So that's really what you think?",
    "I do.",
    "Mmmmm.",
    "Answer quickly, please.",
    "That's a bit annoying TBH.",
    "It's related to that other thing I was telling you about.",
    "I'm not sure if I should believe you - there's some inconsistencies in what you "
    "said.",
    "...and?",
    "Uh huh.",
    "Wait, hold on, what?",
    "Let's back up a bit.",
    "Hahaha no.",
    "Would you mind rephrasing that? It didn't make too much sense to me.",
    "You're not like the other bots.",
    "I feel like we got sidetracked here. Could we go back to the original topic of "
    "conversation?",
    "I'm getting a feeling of déjà vu.",
    "Whatever you say next will probably be completely unrelated to what we were "
    "discussing.",
    "Stop trying to derail the conversation.",
    "Let's stay on-topic, please.",
    "What does that even have to do with anything?",
    "That's nonsense.",
]


def _get_random_text() -> Text:
    """Return a random text to be used in a user or bot message.

    Returns:
        Random text message.
    """
    if random.random() > 0.05:
        return random.choice(MESSAGE_TEXTS)

    # Shout it instead!
    # "That's nice." -> "THAT'S NICE!!!"
    return random.choice(MESSAGE_TEXTS).upper().rstrip(".?!") + "!!!"


def _generate_session_start_events(timestamp: counter) -> List[Event]:
    """Generate events corresponding to the start of a session.

    Args:
        timestamp: Timestamp counter to use.

    Returns:
        List of events.
    """
    return [
        ActionExecuted("action_session_start", timestamp=next(timestamp)),
        SessionStarted(timestamp=next(timestamp)),
    ]


def _generate_conversation_events(length: int, sessions: int) -> List[Event]:
    """Generates a list of conversation events, with random user/bot messages.

    Args:
        length: Number of user + bot messages to generate.
        sessions: Number of sessions to use.

    Returns:
        List of generated events.
    """
    timestamp = counter(start=time.time(), step=1)

    events = []
    messages_per_session = length // sessions
    last_is_user = True

    for i in range(length):
        # Only consider adding a user message if last event was a bot message
        # This is because in general conversations consist of >=1 bot messages,
        # followed by a user message, followed by >=1 bot messages, and so on.
        add_user_event = bool(random.getrandbits(1)) and not last_is_user

        if i % messages_per_session == 0:
            events.extend(_generate_session_start_events(timestamp))
            text = "Hey there!"
        else:
            text = _get_random_text()

        if add_user_event:
            # User message
            if last_is_user is None or not last_is_user:
                events.append(
                    ActionExecuted("action_listen", timestamp=next(timestamp))
                )

            events.append(
                UserUttered(
                    text,
                    timestamp=next(timestamp),
                    input_channel="rest",
                    intent={"confidence": 1.0, "name": "an_intent"},
                )
            )
        else:
            # Bot message
            if last_is_user is None or last_is_user:
                events.append(
                    ActionExecuted("utter_message", timestamp=next(timestamp))
                )
            events.append(BotUttered(text, timestamp=next(timestamp)))

        last_is_user = add_user_event

    return events


def _generate_conversations(
    store: InMemoryTrackerStore, count: int, length: int, sessions: int
) -> None:
    """Generates conversations and stores them into a given `InMemoryTrackerStore`.

    Args:
        count: Number of conversations to generate.
        length: Number of user + bot messages per conversations.
        sessions: Number of sessions to use per conversation.
    """
    cli_utils.print_info(
        f"Generating {count} conversations, each with length {length}."
    )
    for _ in tqdm(range(count), desc="conversations"):
        sender_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(
            sender_id, _generate_conversation_events(length, sessions)
        )
        store.save(tracker)


def main() -> None:
    """Main function for gen_conversations.py"""
    parser = argparse.ArgumentParser(description="Random conversations generator.")
    parser.add_argument(
        "--count",
        "-c",
        metavar="N",
        type=int,
        help="Number of conversations to generate. (default: 1)",
        default=1,
    )

    parser.add_argument(
        "--length",
        "-l",
        metavar="N",
        type=int,
        help="Length of each conversation (number of user messages + bot messages). "
        "Note that additional events will be added, such as SessionStarted or "
        "ActionExecuted. (default: 50)",
        default=50,
    )

    parser.add_argument(
        "--sessions",
        "-s",
        metavar="N",
        type=int,
        help="Number of sessions to use per conversation. (default: 1)",
        default=1,
    )

    parser.add_argument(
        "--endpoints",
        "-e",
        metavar="<path>",
        help="Endpoint configuration file specifying the tracker store and event "
        "broker. (default: endpoints.yml)",
        default="endpoints.yml",
    )

    args = parser.parse_args()

    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)
    broker = EventBroker.create(endpoints.event_broker)
    store = InMemoryTrackerStore(domain=None)

    _generate_conversations(store, args.count, args.length, args.sessions)
    exporter = Exporter(store, broker, args.endpoints)
    exporter.publish_events()


if __name__ == "__main__":
    main()
