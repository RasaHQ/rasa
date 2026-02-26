import os
import time
import uuid

import pytest

from tests.integration_tests.conftest import (
    get_conversation_tracker,
    send_message_to_rasa_server,
)

DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL")


@pytest.fixture
def test_sender_id() -> str:
    return str(uuid.uuid4())


def extract_domain_from_url(url: str) -> str:
    """Extract the domain from a URL."""
    if "/version" in url:
        return url.split("/version")[0]
    return url


@pytest.mark.timeout(960)
def test_sql_tracker_store_aws_rds_iam_auth_enabled(test_sender_id: str) -> None:
    """Test SQL tracker store with AWS RDS IAM deployment."""
    rasa_domain = extract_domain_from_url(DEPLOYMENT_URL)
    _, bot_responses = send_message_to_rasa_server(
        server_location=rasa_domain,
        message="I want to add a new contact",
        sender_id=test_sender_id,
    )

    assert bot_responses != []
    assert (
        bot_responses[0].get("text") == "What's the handle of the user you want to add?"
    )

    tracker = get_conversation_tracker(rasa_domain, test_sender_id)
    assert tracker is not None
    assert tracker.get("sender_id") == test_sender_id
    assert tracker.get("latest_message").get("text") == "I want to add a new contact"
    assert tracker.get("events") != []

    # test token refresh
    time.sleep(901)  # wait for more than 15 minutes (token expiry time)
    _, bot_responses = send_message_to_rasa_server(
        server_location=rasa_domain,
        message="@john_doe",
        sender_id=test_sender_id,
    )
    assert bot_responses != []
    assert (
        bot_responses[0].get("text") == "What's the name of the user you want to add?"
    )
    tracker = get_conversation_tracker(rasa_domain, test_sender_id)
    assert tracker is not None
    assert tracker.get("sender_id") == test_sender_id
    assert tracker.get("latest_message").get("text") == "@john_doe"
    assert tracker.get("events") != []
