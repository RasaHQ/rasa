from tests.integration_tests.conftest import send_message_to_rasa_server


def test_send_message():
    # Send a message and receive a response
    sender_id, response = send_message_to_rasa_server(
        server_location="http://localhost:5005",
        message="I want to transfer money!",
    )

    # Assert that the bot responded
    assert len(response) == 1
    response = response[0]
    assert response["recipient_id"] == sender_id
    assert response["text"] == "Who do you want to transfer money to?"
