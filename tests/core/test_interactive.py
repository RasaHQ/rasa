import asyncio
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Text

import pytest
import uuid

from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses
from mock import Mock

import rasa.utils.io
from rasa.core.actions import action
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.channels import UserMessage
from rasa.core.domain import Domain
from rasa.core.events import BotUttered, ActionExecuted
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training import interactive
from rasa.importers.rasa import TrainingDataImporter
from rasa.nlu.training_data import Message
from rasa.nlu.training_data.loading import RASA, MARKDOWN, UNK
from rasa.utils.endpoints import EndpointConfig
from tests import utilities
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS


@pytest.fixture
def mock_endpoint() -> EndpointConfig:
    return EndpointConfig("https://example.com")


@pytest.fixture
def mock_file_importer(
    default_stack_config: Text, default_nlu_data: Text, default_stories_file: Text
):
    domain_path = DEFAULT_DOMAIN_PATH_WITH_SLOTS
    return TrainingDataImporter.load_from_config(
        default_stack_config, domain_path, [default_nlu_data, default_stories_file]
    )


async def test_send_message(mock_endpoint):
    sender_id = uuid.uuid4().hex

    url = f"{mock_endpoint.url}/conversations/{sender_id}/messages"
    with aioresponses() as mocked:
        mocked.post(url, payload={})

        await interactive.send_message(mock_endpoint, sender_id, "Hello")

        r = utilities.latest_request(mocked, "post", url)

        assert r

        expected = {"sender": "user", "text": "Hello", "parse_data": None}

        assert utilities.json_of_latest_request(r) == expected


async def test_request_prediction(mock_endpoint):
    sender_id = uuid.uuid4().hex

    url = f"{mock_endpoint.url}/conversations/{sender_id}/predict"

    with aioresponses() as mocked:
        mocked.post(url, payload={})

        await interactive.request_prediction(mock_endpoint, sender_id)

        assert utilities.latest_request(mocked, "post", url) is not None


def test_bot_output_format():
    message = {
        "event": "bot",
        "text": "Hello!",
        "data": {
            "image": "http://example.com/myimage.png",
            "attachment": "My Attachment",
            "buttons": [
                {"title": "yes", "payload": "/yes"},
                {"title": "no", "payload": "/no", "extra": "extra"},
            ],
            "elements": [
                {
                    "title": "element1",
                    "buttons": [{"title": "button1", "payload": "/button1"}],
                },
                {
                    "title": "element2",
                    "buttons": [{"title": "button2", "payload": "/button2"}],
                },
            ],
            "quick_replies": [
                {
                    "title": "quick_reply1",
                    "buttons": [{"title": "button3", "payload": "/button3"}],
                },
                {
                    "title": "quick_reply2",
                    "buttons": [{"title": "button4", "payload": "/button4"}],
                },
            ],
        },
    }
    from rasa.core.events import Event

    bot_event = Event.from_parameters(message)

    assert isinstance(bot_event, BotUttered)

    formatted = interactive.format_bot_output(bot_event)
    assert formatted == (
        "Hello!\n"
        "Image: http://example.com/myimage.png\n"
        "Attachment: My Attachment\n"
        "Buttons:\n"
        "1: yes (/yes)\n"
        '2: no (/no) - {"extra": "extra"}\n'
        "Type out your own message...\n"
        "Elements:\n"
        '1: element1 - {"buttons": '
        '[{"payload": "/button1", "title": "button1"}]'
        '}\n2: element2 - {"buttons": '
        '[{"payload": "/button2", "title": "button2"}]'
        "}\nQuick replies:\n"
        '1: quick_reply1 - {"buttons": '
        '[{"payload": "/button3", "title": "button3"}'
        ']}\n2: quick_reply2 - {"buttons": '
        '[{"payload": "/button4", "title": "button4"}'
        "]}"
    )


def test_latest_user_message():
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))

    m = interactive.latest_user_message(tracker_json.get("events"))

    assert m is not None
    assert m["event"] == "user"
    assert m["text"] == "/mood_great"


def test_latest_user_message_on_no_events():
    m = interactive.latest_user_message([])

    assert m is None


def test_all_events_before_user_msg():
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(rasa.utils.io.read_file(tracker_dump))
    evts = tracker_json.get("events")

    m = interactive.all_events_before_latest_user_msg(evts)

    assert m is not None
    assert m == evts[:4]


def test_all_events_before_user_msg_on_no_events():
    assert interactive.all_events_before_latest_user_msg([]) == []


async def test_print_history(mock_endpoint):
    tracker_dump = rasa.utils.io.read_file("data/test_trackers/tracker_moodbot.json")

    sender_id = uuid.uuid4().hex

    url = "{}/conversations/{}/tracker?include_events=AFTER_RESTART".format(
        mock_endpoint.url, sender_id
    )
    with aioresponses() as mocked:
        mocked.get(url, body=tracker_dump, headers={"Accept": "application/json"})

        await interactive._print_history(sender_id, mock_endpoint)

        assert utilities.latest_request(mocked, "get", url) is not None


async def test_is_listening_for_messages(mock_endpoint):
    tracker_dump = rasa.utils.io.read_file("data/test_trackers/tracker_moodbot.json")

    sender_id = uuid.uuid4().hex

    url = "{}/conversations/{}/tracker?include_events=APPLIED".format(
        mock_endpoint.url, sender_id
    )
    with aioresponses() as mocked:
        mocked.get(url, body=tracker_dump, headers={"Content-Type": "application/json"})

        is_listening = await interactive.is_listening_for_message(
            sender_id, mock_endpoint
        )

        assert is_listening


def test_splitting_conversation_at_restarts():
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    evts = json.loads(rasa.utils.io.read_file(tracker_dump)).get("events")
    evts_wo_restarts = evts[:]
    evts.insert(2, {"event": "restart"})
    evts.append({"event": "restart"})

    split = interactive._split_conversation_at_restarts(evts)
    assert len(split) == 2
    assert [e for s in split for e in s] == evts_wo_restarts
    assert len(split[0]) == 2
    assert len(split[0]) == 2


def test_as_md_message():
    parse_data = {
        "text": "Hello there rasa.",
        "entities": [{"start": 12, "end": 16, "entity": "name", "value": "rasa"}],
        "intent": {"name": "greeting", "confidence": 0.9},
    }
    md = interactive._as_md_message(parse_data)
    assert md == "Hello there [rasa](name)."


@pytest.mark.parametrize(
    "parse_original, parse_annotated, expected_entities",
    [
        (
            {
                "text": "Hello there rasa, it's me, paula.",
                "entities": [
                    {
                        "start": 12,
                        "end": 16,
                        "entity": "name1",
                        "value": "rasa",
                        "extractor": "batman",
                    }
                ],
                "intent": {"name": "greeting", "confidence": 0.9},
            },
            {
                "text": "Hello there rasa, it's me, paula.",
                "entities": [
                    {"start": 12, "end": 16, "entity": "name1", "value": "rasa"},
                    {"start": 26, "end": 31, "entity": "name2", "value": "paula"},
                ],
                "intent": {"name": "greeting", "confidence": 0.9},
            },
            [
                {
                    "start": 12,
                    "end": 16,
                    "entity": "name1",
                    "value": "rasa",
                    "extractor": "batman",
                },
                {"start": 26, "end": 31, "entity": "name2", "value": "paula"},
            ],
        ),
        (
            {
                "text": "I am flying from Berlin to London.",
                "entities": [
                    {
                        "start": 17,
                        "end": 23,
                        "entity": "location",
                        "role": "from",
                        "value": "Berlin",
                        "extractor": "DIETClassifier",
                    }
                ],
                "intent": {"name": "inform", "confidence": 0.9},
            },
            {
                "text": "I am flying from Berlin to London.",
                "entities": [
                    {
                        "start": 17,
                        "end": 23,
                        "entity": "location",
                        "value": "Berlin",
                        "role": "from",
                    },
                    {
                        "start": 27,
                        "end": 33,
                        "entity": "location",
                        "value": "London",
                        "role": "to",
                    },
                ],
                "intent": {"name": "inform", "confidence": 0.9},
            },
            [
                {
                    "start": 17,
                    "end": 23,
                    "entity": "location",
                    "value": "Berlin",
                    "role": "from",
                },
                {
                    "start": 27,
                    "end": 33,
                    "entity": "location",
                    "value": "London",
                    "role": "to",
                },
            ],
        ),
        (
            {
                "text": "A large pepperoni and a small mushroom.",
                "entities": [
                    {
                        "start": 2,
                        "end": 7,
                        "entity": "size",
                        "group": "1",
                        "value": "large",
                        "extractor": "DIETClassifier",
                    },
                    {
                        "start": 24,
                        "end": 29,
                        "entity": "size",
                        "value": "small",
                        "extractor": "DIETClassifier",
                    },
                ],
                "intent": {"name": "inform", "confidence": 0.9},
            },
            {
                "text": "A large pepperoni and a small mushroom.",
                "entities": [
                    {
                        "start": 2,
                        "end": 7,
                        "entity": "size",
                        "group": "1",
                        "value": "large",
                    },
                    {
                        "start": 8,
                        "end": 17,
                        "entity": "toppings",
                        "group": "1",
                        "value": "pepperoni",
                    },
                    {
                        "start": 30,
                        "end": 38,
                        "entity": "toppings",
                        "group": "1",
                        "value": "mushroom",
                    },
                    {
                        "start": 24,
                        "end": 29,
                        "entity": "size",
                        "group": "2",
                        "value": "small",
                    },
                ],
                "intent": {"name": "inform", "confidence": 0.9},
            },
            [
                {
                    "start": 2,
                    "end": 7,
                    "entity": "size",
                    "group": "1",
                    "value": "large",
                },
                {
                    "start": 8,
                    "end": 17,
                    "entity": "toppings",
                    "group": "1",
                    "value": "pepperoni",
                },
                {
                    "start": 30,
                    "end": 38,
                    "entity": "toppings",
                    "group": "1",
                    "value": "mushroom",
                },
                {
                    "start": 24,
                    "end": 29,
                    "entity": "size",
                    "group": "2",
                    "value": "small",
                },
            ],
        ),
    ],
)
def test__merge_annotated_and_original_entities(
    parse_original: Dict[Text, Any],
    parse_annotated: Dict[Text, Any],
    expected_entities: List[Dict[Text, Any]],
):
    entities = interactive._merge_annotated_and_original_entities(
        parse_annotated, parse_original
    )
    assert entities == expected_entities


def test_validate_user_message():
    parse_data = {
        "text": "Hello there rasa.",
        "parse_data": {
            "entities": [{"start": 12, "end": 16, "entity": "name", "value": "rasa"}],
            "intent": {"name": "greeting", "confidence": 0.9},
        },
    }
    assert interactive._validate_user_regex(parse_data, ["greeting", "goodbye"])
    assert not interactive._validate_user_regex(parse_data, ["goodbye"])


async def test_undo_latest_msg(mock_endpoint):
    tracker_dump = rasa.utils.io.read_file("data/test_trackers/tracker_moodbot.json")

    sender_id = uuid.uuid4().hex

    url = "{}/conversations/{}/tracker?include_events=ALL".format(
        mock_endpoint.url, sender_id
    )
    append_url = "{}/conversations/{}/tracker/events".format(
        mock_endpoint.url, sender_id
    )
    with aioresponses() as mocked:
        mocked.get(url, body=tracker_dump)
        mocked.post(append_url)

        await interactive._undo_latest(sender_id, mock_endpoint)

        r = utilities.latest_request(mocked, "post", append_url)

        assert r

        # this should be the events the interactive call send to the endpoint
        # these events should have the last utterance omitted
        corrected_event = utilities.json_of_latest_request(r)
        assert corrected_event["event"] == "undo"


def test_utter_custom_message():
    test_event = """
      {
      "data": {
        "attachment": null,
        "buttons": null,
        "elements": [
          {
            "a": "b"
          }
        ]
      },
      "event": "bot",
      "text": null,
      "timestamp": 1542649219.331037
    }
    """
    actual = interactive._chat_history_table([json.loads(test_event)])

    assert json.dumps({"a": "b"}) in actual


async def test_interactive_domain_persistence(
    mock_endpoint: EndpointConfig, tmp_path: Path
):
    # Test method interactive._write_domain_to_file

    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = rasa.utils.io.read_json_file(tracker_dump)

    events = tracker_json.get("events", [])

    domain_path = str(tmp_path / "interactive_domain_save.yml")

    url = f"{mock_endpoint.url}/domain"
    with aioresponses() as mocked:
        mocked.get(url, payload={})

        serialised_domain = await interactive.retrieve_domain(mock_endpoint)
        old_domain = Domain.from_dict(serialised_domain)

        interactive._write_domain_to_file(domain_path, events, old_domain)

    saved_domain = rasa.utils.io.read_config_file(domain_path)

    for default_action in action.default_actions():
        assert default_action.name() not in saved_domain["actions"]


async def test_write_domain_to_file_with_form(tmp_path: Path):
    domain_path = str(tmp_path / "domain.yml")
    form_name = "my_form"
    old_domain = Domain.from_yaml(
        f"""
    actions:
    - utter_greet
    - utter_goodbye
    forms:
    - {form_name}
    intents:
    - greet
    """
    )

    events = [ActionExecuted(form_name), ActionExecuted(ACTION_LISTEN_NAME)]
    events = [e.as_dict() for e in events]

    interactive._write_domain_to_file(domain_path, events, old_domain)

    assert set(Domain.from_path(domain_path).action_names) == set(
        old_domain.action_names
    )


async def test_filter_intents_before_save_nlu_file():
    # Test method interactive._filter_messages
    from random import choice

    greet = {"intent": "greet", "text_features": [0.5]}
    goodbye = {"intent": "goodbye", "text_features": [0.5]}
    test_msgs = [Message("How are you?", greet), Message("I am inevitable", goodbye)]

    domain_file = DEFAULT_DOMAIN_PATH_WITH_SLOTS
    domain = Domain.load(domain_file)
    intents = domain.intents

    msgs = test_msgs.copy()
    if intents:
        msgs.append(Message("/" + choice(intents), greet))

    assert test_msgs == interactive._filter_messages(msgs)


@pytest.mark.parametrize(
    "path, expected_format",
    [("bla.json", RASA), ("other.md", MARKDOWN), ("unknown", UNK)],
)
def test_get_nlu_target_format(path: Text, expected_format: Text):
    assert interactive._get_nlu_target_format(path) == expected_format


@pytest.mark.parametrize(
    "trackers, expected_trackers",
    [
        (
            [DialogueStateTracker.from_events("one", [])],
            [deque([]), UserMessage.DEFAULT_SENDER_ID],
        ),
        (
            [
                str(i)
                for i in range(
                    interactive.MAX_NUMBER_OF_TRAINING_STORIES_FOR_VISUALIZATION + 1
                )
            ],
            [UserMessage.DEFAULT_SENDER_ID],
        ),
    ],
)
async def test_initial_plotting_call(
    mock_endpoint: EndpointConfig,
    monkeypatch: MonkeyPatch,
    trackers: List[Text],
    expected_trackers: List[Text],
    mock_file_importer: TrainingDataImporter,
):
    get_training_trackers = Mock(return_value=trackers)
    monkeypatch.setattr(
        interactive, "_get_training_trackers", asyncio.coroutine(get_training_trackers)
    )

    monkeypatch.setattr(interactive.utils, "is_limit_reached", lambda _, __: True)

    plot_trackers = Mock()
    monkeypatch.setattr(interactive, "_plot_trackers", asyncio.coroutine(plot_trackers))

    url = f"{mock_endpoint.url}/domain"
    with aioresponses() as mocked:
        mocked.get(url, payload={})

        await interactive.record_messages(mock_endpoint, mock_file_importer)

    get_training_trackers.assert_called_once()
    plot_trackers.assert_called_once_with(
        expected_trackers, interactive.DEFAULT_STORY_GRAPH_FILE, mock_endpoint
    )


async def test_not_getting_trackers_when_skipping_visualization(
    mock_endpoint: EndpointConfig, monkeypatch: MonkeyPatch
):
    get_trackers = Mock()
    monkeypatch.setattr(interactive, "_get_tracker_events_to_plot", get_trackers)

    monkeypatch.setattr(interactive.utils, "is_limit_reached", lambda _, __: True)

    url = f"{mock_endpoint.url}/domain"
    with aioresponses() as mocked:
        mocked.get(url, payload={})

        await interactive.record_messages(
            mock_endpoint, mock_file_importer, skip_visualization=True
        )

    get_trackers.assert_not_called()
