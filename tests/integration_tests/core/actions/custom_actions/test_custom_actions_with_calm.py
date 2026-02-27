import pytest

from tests.integration_tests.conftest import (
    get_conversation_tracker,
    send_message_to_rasa_server,
)

HTTP_RASA_SERVER = "http://localhost:5010"
HTTPS_RASA_SERVER = "http://localhost:5011"
GRPC_RASA_SERVER = "http://localhost:5012"
GRPC_SSL_RASA_SERVER = "http://localhost:5013"


@pytest.mark.parametrize(
    "server_location",
    [HTTP_RASA_SERVER, HTTPS_RASA_SERVER, GRPC_RASA_SERVER, GRPC_SSL_RASA_SERVER],
)
def test_custom_action_invocation_with_calm_bot(server_location: str) -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_location, message="list contacts"
    )

    assert len(response_messages) == 2
    response_message = response_messages[0]
    assert response_message["recipient_id"] == sender_id
    assert response_message["text"] == "Here are your contacts: John, Jack, Jane."


@pytest.mark.parametrize(
    "server_location",
    [HTTP_RASA_SERVER, HTTPS_RASA_SERVER, GRPC_RASA_SERVER, GRPC_SSL_RASA_SERVER],
)
def test_custom_action_returning_session_ended_terminates_conversation(
    server_location: str,
) -> None:
    """Custom action returning SessionEnded terminates the conversation."""
    # send regular message
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_location, message="list contacts"
    )

    assert len(response_messages) == 2
    response_message = response_messages[0]
    assert response_message["recipient_id"] == sender_id
    assert response_message["text"] == "Here are your contacts: John, Jack, Jane."

    # send message that triggers the list_reminders flow which also ends the session
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_location,
        message="list my reminders",
        sender_id=sender_id,
    )

    # bot responds before the session ends
    assert (
        response_messages[0]["text"]
        == "You have 2 reminders: dentist appointment and team meeting."
    )

    tracker = get_conversation_tracker(server_location, sender_id)
    assert tracker is not None
    events = tracker["events"]
    assert any(evt.get("event") == "session_ended" for evt in events)

    # Sending another message to the same terminated conversation should do nothing
    # (no response, no new events)
    _, response_messages = send_message_to_rasa_server(
        server_location=server_location,
        message="hello",
        sender_id=sender_id,
    )
    assert len(response_messages) == 0

    tracker_after = get_conversation_tracker(server_location, sender_id)
    assert len(tracker_after["events"]) == len(events)
