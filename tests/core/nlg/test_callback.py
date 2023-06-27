from rasa.core.nlg.callback import nlg_request_format
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity


def test_nlg_request_format(
    default_tracker: DialogueStateTracker,
) -> None:
    assert nlg_request_format(
        utter_action="utter_one_id",
        tracker=default_tracker,
        output_channel="test",
    ) == {
        "response": "utter_one_id",
        "id": "",
        "arguments": {},
        "channel": {"name": "test"},
        "tracker": default_tracker.current_state(EventVerbosity.ALL),
    }


def test_nlg_request_format_with_arguments(
    default_tracker: DialogueStateTracker,
) -> None:
    assert nlg_request_format(
        utter_action="utter_one_id",
        tracker=default_tracker,
        output_channel="test",
        some_arg="some_value",
    ) == {
        "response": "utter_one_id",
        "id": "",
        "arguments": {"some_arg": "some_value"},
        "channel": {"name": "test"},
        "tracker": default_tracker.current_state(EventVerbosity.ALL),
    }


def test_nlg_request_format_with_response_ids(
    default_tracker: DialogueStateTracker,
) -> None:
    assert nlg_request_format(
        utter_action="utter_one_id",
        tracker=default_tracker,
        output_channel="test",
        response_id="1",
    ) == {
        "response": "utter_one_id",
        "id": "1",
        "arguments": {},
        "channel": {"name": "test"},
        "tracker": default_tracker.current_state(EventVerbosity.ALL),
    }
