import pytest

from tests.integration_tests.conftest import send_message_to_rasa_server

HTTP_RASA_SERVER = "http://localhost:5010"
HTTPS_RASA_SERVER = "http://localhost:5011"
GRPC_RASA_SERVER = "http://localhost:5012"
GRPC_SSL_RASA_SERVER = "http://localhost:5013"


@pytest.mark.parametrize(
    "server_location",
    [HTTP_RASA_SERVER, HTTPS_RASA_SERVER, GRPC_RASA_SERVER, GRPC_SSL_RASA_SERVER],
)
def test_custom_action_invocation_with_nlu_bot(server_location: str) -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_location, message="cu"
    )

    assert len(response_messages) == 1
    response_message = response_messages[0]
    assert response_message["recipient_id"] == sender_id
    assert response_message["text"] == "Goodbye!"


@pytest.mark.parametrize(
    "server_location",
    [HTTP_RASA_SERVER, HTTPS_RASA_SERVER, GRPC_RASA_SERVER, GRPC_SSL_RASA_SERVER],
)
def test_custom_validation_action_invocation_with_nlu_bot_negative_mood_intent(
    server_location: str,
) -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_location, message="I am so sad"
    )

    assert len(response_messages) == 2
    response_text_message = response_messages[0]
    assert response_text_message["recipient_id"] == sender_id
    assert response_text_message["text"] == "Here is something to cheer you up:"

    response_image_message = response_messages[1]
    assert response_image_message["image"] == "https://i.imgur.com/nGF1K8f.jpg"


@pytest.mark.parametrize(
    "server_location",
    [HTTP_RASA_SERVER, HTTPS_RASA_SERVER, GRPC_RASA_SERVER, GRPC_SSL_RASA_SERVER],
)
def test_custom_validation_action_invocation_with_nlu_bot_positive_mood_intent(
    server_location: str,
) -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_location, message="I am feeling great"
    )

    assert len(response_messages) == 1
    response_message = response_messages[0]
    assert response_message["recipient_id"] == sender_id
    assert response_message["text"] == "Great, carry on!"
