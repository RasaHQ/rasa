from tests.integration_tests.conftest import send_message_to_rasa_server

GRPC_RASA_SERVER = "http://localhost:5010"
GRPC_SSL_RASA_SERVER = "http://localhost:5011"


def test_grpc_custom_action_invocation() -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=GRPC_RASA_SERVER, message="cu"
    )

    assert len(response_messages) == 1
    response_message = response_messages[0]
    assert response_message["recipient_id"] == sender_id
    assert response_message["text"] == "Goodbye!"


def test_grpc_ssl_custom_action_invocation() -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=GRPC_SSL_RASA_SERVER, message="cu"
    )

    assert len(response_messages) == 1
    response_message = response_messages[0]
    assert response_message["recipient_id"] == sender_id
    assert response_message["text"] == "Goodbye!"
