import json
import pytest
import uuid
from aioresponses import aioresponses

import rasa.utils.io
from rasa.core.events import BotUttered
from rasa.core.training import interactive
from rasa.utils.endpoints import EndpointConfig
from rasa.core.actions.action import default_actions
from rasa.core.domain import Domain
from tests.utilities import latest_request, json_of_latest_request


@pytest.fixture
def mock_endpoint():
    return EndpointConfig("https://example.com")


async def test_send_message(mock_endpoint):
    sender_id = uuid.uuid4().hex

    url = "{}/conversations/{}/messages".format(mock_endpoint.url, sender_id)
    with aioresponses() as mocked:
        mocked.post(url, payload={})

        await interactive.send_message(mock_endpoint, sender_id, "Hello")

        r = latest_request(mocked, "post", url)

        assert r

        expected = {"sender": "user", "text": "Hello", "parse_data": None}

        assert json_of_latest_request(r) == expected


async def test_request_prediction(mock_endpoint):
    sender_id = uuid.uuid4().hex

    url = "{}/conversations/{}/predict".format(mock_endpoint.url, sender_id)

    with aioresponses() as mocked:
        mocked.post(url, payload={})

        await interactive.request_prediction(mock_endpoint, sender_id)

        assert latest_request(mocked, "post", url) is not None


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

        assert latest_request(mocked, "get", url) is not None


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


def test_entity_annotation_merge_with_original():
    parse_original = {
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
    }
    parse_annotated = {
        "text": "Hello there rasa, it's me, paula.",
        "entities": [
            {"start": 12, "end": 16, "entity": "name1", "value": "rasa"},
            {"start": 26, "end": 31, "entity": "name2", "value": "paula"},
        ],
        "intent": {"name": "greeting", "confidence": 0.9},
    }

    entities = interactive._merge_annotated_and_original_entities(
        parse_annotated, parse_original
    )
    assert entities == [
        {
            "start": 12,
            "end": 16,
            "entity": "name1",
            "value": "rasa",
            "extractor": "batman",
        },
        {"start": 26, "end": 31, "entity": "name2", "value": "paula"},
    ]


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

        r = latest_request(mocked, "post", append_url)

        assert r

        # this should be the events the interactive call send to the endpoint
        # these events should have the last utterance omitted
        corrected_event = json_of_latest_request(r)
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


async def test_interactive_domain_persistence(mock_endpoint, tmpdir):
    # Test method interactive._write_domain_to_file

    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = rasa.utils.io.read_json_file(tracker_dump)

    events = tracker_json.get("events", [])

    domain_path = tmpdir.join("interactive_domain_save.yml").strpath

    url = "{}/domain".format(mock_endpoint.url)
    with aioresponses() as mocked:
        mocked.get(url, payload={})

        serialised_domain = await interactive.retrieve_domain(mock_endpoint)
        old_domain = Domain.from_dict(serialised_domain)

        await interactive._write_domain_to_file(domain_path, events, old_domain)

    saved_domain = rasa.utils.io.read_config_file(domain_path)

    for default_action in default_actions():
        assert default_action.name() not in saved_domain["actions"]
