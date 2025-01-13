from typing import Any, Dict, List, Text
from unittest.mock import AsyncMock, _Call, call

import pytest

from rasa.core.channels.socketio import SocketIOOutput
from rasa.shared.core.trackers import DialogueStateTracker


@pytest.fixture
def sample_events():
    return [
        {
            "event": "slot",
            "timestamp": 1727272173.897743,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "session_started_metadata",
            "value": {
                "call_id": "de0c2968-0840-458d-b2d2-70d17183e41f",
                "user_phone": "+491604697810",
                "bot_phone": "+493041733972",
                "user_name": None,
                "user_host": None,
                "bot_host": None,
                "direction": None,
            },
        },
        {
            "event": "action",
            "timestamp": 1727272173.96827,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_session_start",
            "policy": None,
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "slot",
            "timestamp": 1727272173.968357,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "user_phone",
            "value": "+491604697810",
        },
        {
            "event": "slot",
            "timestamp": 1727272173.968363,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "bot_phone",
            "value": "+493041733972",
        },
        {
            "event": "user",
            "timestamp": 1727272173.971586,
            "metadata": {
                "call_id": "de0c2968-0840-458d-b2d2-70d17183e41f",
                "user_phone": "+491604697810",
                "bot_phone": "+493041733972",
                "user_name": None,
                "user_host": None,
                "bot_host": None,
                "direction": None,
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "/session_start",
            "parse_data": {
                "intent": {"name": "session_start", "confidence": 1.0},
                "entities": [],
                "text": "/session_start",
                "message_id": "a5185844d27247d0bf48d56b176be668",
                "metadata": {
                    "call_id": "de0c2968-0840-458d-b2d2-70d17183e41f",
                    "user_phone": "+491604697810",
                    "bot_phone": "+493041733972",
                    "user_name": None,
                    "user_host": None,
                    "bot_host": None,
                    "direction": None,
                },
                "intent_ranking": [{"name": "session_start", "confidence": 1.0}],
                "commands": [{"command": "session start"}],
            },
            "input_channel": None,
            "message_id": "a5185844d27247d0bf48d56b176be668",
        },
        {
            "event": "slot",
            "timestamp": 1727272173.979328,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "flow_hashes",
            "value": {
                "add_card": "4539d3a461b81c280996ccbfd7c08961",
                "add_contact": "3a53ed66f48107f368863f8f43868f8e",
                "authenticate_user": "f0bb9d7c4f406adf9513f62cfe5e93e6",
                "check_balance": "654ddd6d89422ec03508df5b26013952",
                "check_portfolio": "a073b01d916f96d71fb66bb4882e2b65",
                "list_contacts": "e730fa1fa90b8d589d0b16da248fc9de",
                "pattern_correction": "e2e11bb69b36d71922ebe4f4e4d6bd09",
                "pattern_chitchat": "098e668f52b891bf27850529a2e945b8",
                "pattern_search": "17455f745ae7ee1df7fc753c0edec7d6",
                "pattern_cancel_flow": "590512da2aa7a959ca575ba0bd637e80",
                "pattern_completed": "7b7f7b743f58feac88fecc3f33cfc27f",
                "pattern_session_start": "29f1b3cbbc70f2eed5d28c977c86f9ad",
                "register_to_vote_in_california": "ed40621efc016de8a80c39e1167fec19",
                "remove_contact": "d4fef4f673ed9c8ca60d238f90e7f97a",
                "replace_card": "abcbbd4b4a6ffccc3ad688ce73be4505",
                "replace_eligible_card": "87712cffed61dd6608100a08a8441b40",
                "setup_recurrent_payment": "d4f248dc150ab6c4cc44bab2e7e90099",
                "transaction_search": "e991ffe99f99f6636012a217abfa914d",
                "transfer_money": "e74cc2d03018e539911d9c3a191f2539",
                "verify_account": "36e8858512cf1e894a123bd8c708b662",
                "whoami": "37b3d62249f8672b6d19d4e0bb1649db",
                "pattern_cannot_handle": "d1f8d1762a5510c4866d5d8ecb2847e7",
                "pattern_clarification": "4462d7e0d2ab52270711d4feb9bbd8eb",
                "pattern_code_change": "b91145d7d9b528c04ff82e7865a8cae5",
                "pattern_collect_information": "28816a73d844a2b866d67659dd661260",
                "pattern_continue_interrupted": "34ac18f3c20c28450b91b25d7318b12c",
                "pattern_human_handoff": "b01db228e6a39f5f8f4acc863b5607cd",
                "pattern_internal_error": "a534aab9a8c3b94876a46f3e098a2950",
                "pattern_restart": "33039d78f669472478d39409f0a69464",
                "pattern_skip_question": "439851a406df7c142ec9e054c95562d0",
            },
        },
        {
            "event": "stack",
            "timestamp": 1727272173.9793322,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "add", "path": "/0", "value": {"frame_id": "ZDDUSEJI", "flow_id": "pattern_session_start", "step_id": "START", "type": "pattern_session_start"}}]',  # noqa E501
        },
        {
            "event": "stack",
            "timestamp": 1727272173.989412,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/0/step_id", "value": "0_utter_greet"}]',  # noqa E501
        },
        {
            "event": "flow_started",
            "timestamp": 1727272173.989456,
            "metadata": {
                "frame_id": "ZDDUSEJI",
                "flow_id": "pattern_session_start",
                "step_id": "0_utter_greet",
                "type": "pattern_session_start",
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_session_start",
        },
        {
            "event": "action",
            "timestamp": 1727272173.989463,
            "metadata": {
                "active_flow": "pattern_session_start",
                "step_id": "0_utter_greet",
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "utter_greet",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
        {
            "event": "bot",
            "timestamp": 1727272173.989652,
            "metadata": {
                "active_flow": "pattern_session_start",
                "step_id": "0_utter_greet",
                "utter_action": "utter_greet",
                "utter_source": "TemplatedNaturalLanguageGenerator",
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "text": "Hi! I'm your Financial Assistant! How can I help you?",
            "data": {
                "elements": None,
                "quick_replies": None,
                "buttons": None,
                "attachment": None,
                "image": None,
                "custom": None,
            },
        },
        {
            "event": "stack",
            "timestamp": 1727272173.99662,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "replace", "path": "/0/step_id", "value": "END"}]',
        },
        {
            "event": "stack",
            "timestamp": 1727272173.996663,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "update": '[{"op": "remove", "path": "/0"}]',
        },
        {
            "event": "flow_completed",
            "timestamp": 1727272173.99669,
            "metadata": {
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "flow_id": "pattern_session_start",
            "step_id": "0_utter_greet",
        },
        {
            "event": "action",
            "timestamp": 1727272173.996696,
            "metadata": {
                "active_flow": None,
                "step_id": None,
                "model_id": "2fbbd6456437492c92a5079c90ecb33b",
                "assistant_id": "20240418-073244-narrow-archive",
            },
            "name": "action_listen",
            "policy": "FlowPolicy",
            "confidence": 1.0,
            "action_text": None,
            "hide_rule_turn": False,
        },
    ]


@pytest.fixture
def socketio_output(default_tracker: DialogueStateTracker):
    sio = AsyncMock()
    output_channel = SocketIOOutput(sio, "bot")
    output_channel.attach_tracker_state(default_tracker)
    return output_channel


@pytest.fixture
def expected_tracker_state_call():
    return call(
        "tracker_state",
        {
            "sender_id": "my-sender",
            "slots": {
                "name": None,
                "requested_slot": None,
                "flow_hashes": None,
                "session_started_metadata": None,
            },
            "latest_message": {
                "intent": {},
                "entities": [],
                "text": None,
                "message_id": None,
                "metadata": {},
            },
            "latest_event_time": None,
            "followup_action": "action_listen",
            "paused": False,
            "stack": [],
            "events": [],
            "latest_input_channel": None,
            "active_loop": {},
            "latest_action": {},
            "latest_action_name": None,
        },
        room="recipient_id",
    )


async def test_socketio_handles_buttons_without_payload(
    socketio_output: SocketIOOutput, expected_tracker_state_call: _Call
):
    message = {
        "text": "hello world",
        "buttons": [{"title": "Button1"}],
    }

    # Send the message
    await socketio_output.send_response("recipient_id", message)

    # Check if the socketio object was called with the correct arguments
    expected_calls = [
        expected_tracker_state_call,
        call(
            "bot",
            {
                "text": "hello world",
                "quick_replies": [
                    {"content_type": "text", "title": "Button1", "payload": "Button1"}
                ],
            },
            room="recipient_id",
        ),
    ]
    socketio_output.sio.emit.assert_has_calls(expected_calls, any_order=False)


async def test_socketio_handles_buttons_with_payload(
    socketio_output: SocketIOOutput, expected_tracker_state_call: _Call
):
    message = {
        "text": "hello world",
        "buttons": [{"title": "Button1", "payload": "/example_intent"}],
    }

    # Send the message
    await socketio_output.send_response("recipient_id", message)

    # Check if the socketio object was called with the correct arguments
    expected_calls = [
        expected_tracker_state_call,
        call(
            "bot",
            {
                "text": "hello world",
                "quick_replies": [
                    {
                        "content_type": "text",
                        "title": "Button1",
                        "payload": "/example_intent",
                    }
                ],
            },
            room="recipient_id",
        ),
    ]
    socketio_output.sio.emit.assert_has_calls(expected_calls, any_order=False)


def test_get_new_events(
    socketio_output: SocketIOOutput, sample_events: List[Dict[Text, Any]]
):
    # Set up the tracker state with our sample events
    socketio_output.tracker_state = {"events": sample_events}

    # First call should return all events
    first_call_events = socketio_output._get_new_events()
    assert len(first_call_events) == len(sample_events)
    assert socketio_output.last_event_timestamp == sample_events[-1]["timestamp"]

    # Second call should return no events
    second_call_events = socketio_output._get_new_events()
    assert len(second_call_events) == 0

    # Add a new event and check if it's returned
    new_event = {"event": "user", "timestamp": 1727272174.0, "text": "Hello"}
    socketio_output.tracker_state["events"].append(new_event)
    third_call_events = socketio_output._get_new_events()
    assert len(third_call_events) == 1
    assert third_call_events[0] == new_event
    assert socketio_output.last_event_timestamp == new_event["timestamp"]

    # Final call should return no events
    fourth_call_events = socketio_output._get_new_events()
    assert len(fourth_call_events) == 0
