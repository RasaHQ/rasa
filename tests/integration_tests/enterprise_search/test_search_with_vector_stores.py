import pytest
from tests.integration_tests.conftest import send_message_to_rasa_server

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
def test_enterprise_search_with_vector_store(server_url):
    sender_id, response_messages = send_message_to_rasa_server(
        server_location=server_url,
        message="Tell me about the weekly discounts and exclusive offers",
    )
    response_message = response_messages[0]
    assert len(response_messages) == 2
    assert response_message["recipient_id"] == sender_id
    assert "weekly discounts and exclusive offers" in response_message["text"]
