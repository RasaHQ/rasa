import json
import pytest
import uuid
from httpretty import httpretty

from rasa_core import utils
from rasa_core.training import interactive
from rasa_core.utils import EndpointConfig


@pytest.fixture
def mock_endpoint():
    return EndpointConfig("https://abc.defg")


def test_send_message(mock_endpoint):
    sender_id = uuid.uuid4().hex

    url = '{}/conversations/{}/messages'.format(
        mock_endpoint.url, sender_id)
    httpretty.register_uri(httpretty.POST, url, body='{}')

    httpretty.enable()
    interactive.send_message(mock_endpoint, sender_id, "Hello")
    httpretty.disable()

    b = httpretty.latest_requests[-1].body.decode("utf-8")
    assert json.loads(b) == {
        "sender": "user",
        "message": "Hello",
        "parse_data": None
    }


def test_request_prediction(mock_endpoint):
    sender_id = uuid.uuid4().hex

    url = '{}/conversations/{}/predict'.format(
        mock_endpoint.url, sender_id)
    httpretty.register_uri(httpretty.POST, url, body='{}')

    httpretty.enable()
    interactive.request_prediction(mock_endpoint, sender_id)
    httpretty.disable()

    b = httpretty.latest_requests[-1].body.decode("utf-8")
    assert b == ""


def test_bot_output_format():
    message = {
        "text": "Hello!",
        "data": {
            "image": "http://example.com/myimage.png",
            "attachment": "My Attachment",
            "buttons": [
                {"title": "yes", "payload": "/yes"},
                {"title": "no", "payload": "/no"}]
        }
    }
    formatted = interactive.format_bot_output(message)
    assert formatted == ("Hello!\n"
                         "Image: http://example.com/myimage.png\n"
                         "Attachment: My Attachment\n"
                         "1: yes (/yes)\n"
                         "2: no (/no)")


def test_latest_user_message():
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))

    m = interactive.latest_user_message(tracker_json.get("events"))

    assert m is not None
    assert m["event"] == "user"
    assert m["text"] == "/mood_great"


def test_latest_user_message_on_no_events():
    m = interactive.latest_user_message([])

    assert m is None


def test_all_events_before_user_msg():
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    tracker_json = json.loads(utils.read_file(tracker_dump))
    evts = tracker_json.get("events")

    m = interactive.all_events_before_latest_user_msg(evts)

    assert m is not None
    assert m == evts[:4]


def test_all_events_before_user_msg_on_no_events():
    assert interactive.all_events_before_latest_user_msg([]) == []


def test_print_history(mock_endpoint):
    tracker_dump = utils.read_file(
        "data/test_trackers/tracker_moodbot.json")

    sender_id = uuid.uuid4().hex

    url = '{}/conversations/{}/tracker'.format(
        mock_endpoint.url, sender_id)
    httpretty.register_uri(httpretty.GET, url, body=tracker_dump)

    httpretty.enable()
    interactive._print_history(sender_id, mock_endpoint)
    httpretty.disable()

    b = httpretty.latest_requests[-1].body.decode("utf-8")
    assert b == ""
    assert (httpretty.latest_requests[-1].path ==
            "/conversations/{}/tracker?include_events=AFTER_RESTART"
            "".format(sender_id))


def test_is_listening_for_messages(mock_endpoint):
    tracker_dump = utils.read_file(
        "data/test_trackers/tracker_moodbot.json")

    sender_id = uuid.uuid4().hex

    url = '{}/conversations/{}/tracker'.format(
        mock_endpoint.url, sender_id)
    httpretty.register_uri(httpretty.GET, url, body=tracker_dump)

    httpretty.enable()
    is_listening = interactive.is_listening_for_message(sender_id,
                                                        mock_endpoint)
    httpretty.disable()

    assert is_listening


def test_splitting_conversation_at_restarts():
    tracker_dump = "data/test_trackers/tracker_moodbot.json"
    evts = json.loads(utils.read_file(tracker_dump)).get("events")
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
        "entities": [{"start": 12,
                      "end": 16,
                      "entity": "name",
                      "value": "rasa"}],
        "intent": {"name": "greeting", "confidence": 0.9}
    }
    md = interactive._as_md_message(parse_data)
    assert md == "Hello there [rasa](name)."


def test_validate_user_message():
    parse_data = {
        "text": "Hello there rasa.",
        "parse_data": {
            "entities": [{"start": 12,
                          "end": 16,
                          "entity": "name",
                          "value": "rasa"}],
            "intent": {"name": "greeting", "confidence": 0.9}
        }
    }
    assert interactive._validate_user_regex(parse_data, ["greeting", "goodbye"])
    assert not interactive._validate_user_regex(parse_data, ["goodbye"])


def test_undo_latest_msg(mock_endpoint):
    tracker_dump = utils.read_file(
        "data/test_trackers/tracker_moodbot.json")
    tracker_json = json.loads(tracker_dump)
    evts = tracker_json.get("events")

    sender_id = uuid.uuid4().hex

    url = '{}/conversations/{}/tracker'.format(
        mock_endpoint.url, sender_id)
    replace_url = '{}/conversations/{}/tracker/events'.format(
        mock_endpoint.url, sender_id)
    httpretty.register_uri(httpretty.GET, url, body=tracker_dump)
    httpretty.register_uri(httpretty.PUT, replace_url)

    httpretty.enable()
    interactive._undo_latest(sender_id, mock_endpoint)
    httpretty.disable()

    b = httpretty.latest_requests[-1].body.decode("utf-8")

    # this should be the events the interactive call send to the endpoint
    # these events should have the last utterance omitted
    replaced_evts = json.loads(b)
    assert len(replaced_evts) == 6
    assert replaced_evts == evts[:6]
