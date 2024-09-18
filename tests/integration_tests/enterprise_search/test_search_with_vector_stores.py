import pytest
from tests.integration_tests.conftest import (
    send_message_to_rasa_server,
    get_conversation_tracker,
    was_enterprise_search_policy_used,
)

RASA_SERVER_TEST_URL_FAISS_CONFIG = "http://localhost:5005"
RASA_SERVER_TEST_URL_QDRANT_CONFIG = "http://localhost:5007"
RASA_SERVER_TEST_URL_MILVUS_CONFIG = "http://localhost:5006"


@pytest.mark.parametrize(
    "server_url",
    [
        RASA_SERVER_TEST_URL_FAISS_CONFIG,
        RASA_SERVER_TEST_URL_QDRANT_CONFIG,
        RASA_SERVER_TEST_URL_MILVUS_CONFIG,
    ],
)
def test_enterprise_search_with_vector_store(server_url: str) -> None:
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_url,
        message="Can I book restaurants, hotels, and flights through FinX?",
    )

    assert len(response_messages) == 2
    response_msg = response_messages[0]
    assert response_msg["recipient_id"] == sender_id
    assert "yes" in response_msg["text"].lower()


@pytest.mark.parametrize(
    "query",
    [
        "How can I contact FinX customer support?",
        "Can I use FinX for international transfers?",
    ],
)
def test_enterprise_search_policy_invoked(query: str) -> None:
    """Test that the EnterpriseSearchPolicy is used to answer a given query."""
    RASA_SERVER_TEST_URL = RASA_SERVER_TEST_URL_FAISS_CONFIG
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=RASA_SERVER_TEST_URL,
        message=query,
    )

    tracker_json = get_conversation_tracker(
        server_location=RASA_SERVER_TEST_URL,
        conversation_id=sender_id,
    )

    assert tracker_json is not None, "Failed to retrieve tracker JSON"
    assert len(response_messages) == 2, "Expected 2 response messages"
    response_text = response_messages[0]["text"]
    assert was_enterprise_search_policy_used(
        tracker_json
    ), f"EnterpriseSearchPolicy was not used for response: {response_text}"


@pytest.mark.parametrize(
    "query",
    [
        "what is my balance?",
        "send money to mom",
    ],
)
def test_enterprise_search_policy_not_invoked(query: str) -> None:
    """Test that the EnterpriseSearchPolicy is NOT used to answer a given query."""
    RASA_SERVER_TEST_URL = RASA_SERVER_TEST_URL_FAISS_CONFIG
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=RASA_SERVER_TEST_URL,
        message=query,
    )

    tracker_json = get_conversation_tracker(
        server_location=RASA_SERVER_TEST_URL,
        conversation_id=sender_id,
    )

    assert tracker_json is not None, "Failed to retrieve tracker JSON"
    response_text = response_messages[0]["text"]
    assert (
        was_enterprise_search_policy_used(tracker_json) is False
    ), f"EnterpriseSearchPolicy was used for response: {response_text}"


@pytest.mark.parametrize(
    "server_url",
    [
        RASA_SERVER_TEST_URL_QDRANT_CONFIG,
        RASA_SERVER_TEST_URL_MILVUS_CONFIG,
    ],
)
def test_response_pattern_cannot_handle(server_url: str) -> None:
    query = "Who is the senator of Wyoming?"
    _, response_messages = send_message_to_rasa_server(
        server_location=server_url,
        message=query,
    )

    # pattern cannot handle is triggered
    assert response_messages is not None, "Failed to retrieve response messages"
    assert (
        response_messages[0]["text"]
        == "I didn't quite understand that. Can you rephrase?"
    )
