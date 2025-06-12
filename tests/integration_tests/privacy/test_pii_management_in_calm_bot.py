import json
import time
from typing import Any, Dict, Generator, List

import pytest
import requests
from confluent_kafka import Consumer

from tests.integration_tests.conftest import send_message_to_rasa_server

HTTP_RASA_SERVER_ANONYMIZE = "http://localhost:5005"
HTTP_RASA_SERVER_DELETE = "http://localhost:5006"

USER_MESSAGES = [
    "I want to make a tax payment.",
    "My name is John Doe.",
    "My national insurance number is AB123456C.",
    "My credit card number is 1234-5678-9012-3456.",
    "I've had a great experience, thank you! "
    "Shame that you are not able to process my direct debit "
    "payments from my bank account number 2715500356.",
]

BOT_MESSAGES = [
    {-1: "What is your full name?"},
    {-1: "What is your National Insurance Number?"},
    {-1: "What is your credit card number?"},
    {
        0: "Your payment has been submitted successfully with the following details:",
        1: "Full Name: JOHN DOE\nNational Insurance Number: AB123456C\n"
        "Credit Card Number: 1234-5678-9012-3456",
        2: "Please provide your feedback on the service.",
    },
    {},
]


@pytest.fixture
def kafka_config() -> Dict[str, Any]:
    return {
        "bootstrap.servers": "localhost:9092",
        "group.id": "pii_test_group",
        "auto.offset.reset": "earliest",
        "security.protocol": "PLAINTEXT",
    }


@pytest.fixture
def test_kafka_consumer(
    kafka_config: Dict[str, Any],
) -> Generator[Consumer, None, None]:
    consumer = Consumer(kafka_config)
    yield consumer
    consumer.close()


def consume_all_messages(consumer: Consumer, topic: str) -> List[Dict[str, Any]]:
    consumer.subscribe([topic])
    messages: List[Dict[str, Any]] = []

    timeout_count = 0
    max_timeout_count = 5
    while timeout_count <= max_timeout_count:
        message = consumer.poll(timeout=1.0)
        if message is None:
            timeout_count += 1
            continue
        if message.error():
            raise Exception(f"Kafka Error: {message.error()}")
        messages.append(
            {
                "value": json.loads(message.value().decode("utf-8")),
                "topic": message.topic(),
                "partition": message.partition(),
            }
        )
        timeout_count = 0
    consumer.unsubscribe()
    return messages


def send_user_messages_to_rasa_pro(server_location: str) -> str:
    sender_id = None

    for user_message, bot_message in zip(USER_MESSAGES, BOT_MESSAGES):
        sender_id, response_messages = send_message_to_rasa_server(
            server_location=server_location, message=user_message, sender_id=sender_id
        )
        for bot_response_index, bot_response_text in bot_message.items():
            assert (
                response_messages[bot_response_index].get("text") == bot_response_text
            )

    return sender_id


def retrieve_tracker_for_sender_id(
    server_location: str, sender_id: str
) -> Dict[str, Any]:
    tracker_url = f"{server_location}/conversations/{sender_id}/tracker"
    response = requests.get(tracker_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to retrieve tracker for sender_id {sender_id}: {response.text}. "
            f"Response code: {response.status_code}"
        )


@pytest.mark.timeout(300)
def test_pii_management_in_calm_bot_anonymization(
    test_kafka_consumer: Consumer,
) -> None:
    """Test the anonymization of PII data in two distinct scenarios.

    First, test the anonymization of PII in events published to the Kafka event broker.
    Second, test the anonymization of PII in trackers stored in a SQL tracker store.
    Also tests that PII slots with custom validation are anonymized correctly in all
    3 supported event types.
    """
    sender_id = send_user_messages_to_rasa_pro(HTTP_RASA_SERVER_ANONYMIZE)

    # test that anonymized events are published to the Kafka event broker
    anonymization_messages = consume_all_messages(test_kafka_consumer, "anonymization")
    assert len(anonymization_messages) > 0

    user_messages = list(
        filter(lambda m: m["value"]["event"] == "user", anonymization_messages)
    )
    assert user_messages[1]["value"]["text"] == "My name is [FULL_NAME]."
    assert user_messages[1]["value"]["parse_data"]["text"] == "My name is [FULL_NAME]."
    assert (
        user_messages[2]["value"]["text"]
        == "My national insurance number is [NATIONAL_INSURANCE_NUMBER]."
    )
    assert (
        user_messages[2]["value"]["parse_data"]["text"]
        == "My national insurance number is [NATIONAL_INSURANCE_NUMBER]."
    )
    assert (
        user_messages[3]["value"]["text"]
        == "My credit card number is ***************3456."
    )
    assert (
        user_messages[3]["value"]["parse_data"]["text"]
        == "My credit card number is ***************3456."
    )
    assert (
        user_messages[4]["value"]["text"] == "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account "
        "number 2715500356."
    )
    assert (
        user_messages[4]["value"]["parse_data"]["text"]
        == "I've had a great experience, thank you! "
        "Shame that you are not able to process "
        "my direct debit payments from my bank "
        "account number 2715500356."
    )

    slot_events = list(
        filter(lambda m: m["value"]["event"] == "slot", anonymization_messages)
    )
    assert slot_events[2]["value"]["name"] == "full_name"
    assert slot_events[2]["value"]["value"] == "[FULL_NAME]"
    # this is the validated slot event for full_name
    assert slot_events[3]["value"]["name"] == "full_name"
    assert slot_events[3]["value"]["value"] == "[FULL_NAME]"
    assert slot_events[4]["value"]["name"] == "national_insurance_number"
    assert slot_events[4]["value"]["value"] == "[NATIONAL_INSURANCE_NUMBER]"
    assert slot_events[5]["value"]["name"] == "credit_card_number"
    assert slot_events[5]["value"]["value"] == "***************3456"

    bot_events = list(
        filter(lambda m: m["value"]["event"] == "bot", anonymization_messages)
    )
    assert (
        bot_events[4]["value"]["text"]
        == "Your payment has been submitted successfully with the "
        "following details:\n\nFull Name: [FULL_NAME]\nNational "
        "Insurance Number: [NATIONAL_INSURANCE_NUMBER]\nCredit "
        "Card Number: ***************3456"
    )

    rasa_messages = consume_all_messages(test_kafka_consumer, "rasa")
    # stream_pii for event broker is set to False,
    # so we expect no messages to this topic
    assert len(rasa_messages) == 0

    time.sleep(120)  # wait for anonymization job to complete

    tracker = retrieve_tracker_for_sender_id(HTTP_RASA_SERVER_ANONYMIZE, sender_id)
    tracker_events = tracker.get("events", [])

    # test that anonymized events are stored in the SQL tracker store
    tracker_user_events = list(filter(lambda m: m["event"] == "user", tracker_events))
    assert tracker_user_events[1]["text"] == "My name is [FULL_NAME]."
    assert (
        tracker_user_events[2]["text"]
        == "My national insurance number is [NATIONAL_INSURANCE_NUMBER]."
    )
    assert (
        tracker_user_events[3]["text"]
        == "My credit card number is ***************3456."
    )
    assert (
        tracker_user_events[4]["text"] == "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account number 2715500356."
    )

    tracker_slot_events = list(filter(lambda m: m["event"] == "slot", tracker_events))
    assert tracker_slot_events[2]["name"] == "full_name"
    assert tracker_slot_events[2]["value"] == "[FULL_NAME]"
    # this is the validated slot event for full_name
    assert tracker_slot_events[3]["name"] == "full_name"
    assert tracker_slot_events[3]["value"] == "[FULL_NAME]"
    assert tracker_slot_events[4]["name"] == "national_insurance_number"
    assert tracker_slot_events[4]["value"] == "[NATIONAL_INSURANCE_NUMBER]"
    assert tracker_slot_events[5]["name"] == "credit_card_number"
    assert tracker_slot_events[5]["value"] == "***************3456"

    tracker_bot_events = list(filter(lambda m: m["event"] == "bot", tracker_events))
    assert (
        tracker_bot_events[4]["text"]
        == "Your payment has been submitted successfully with the following "
        "details:\n\nFull Name: [FULL_NAME]\nNational Insurance Number: "
        "[NATIONAL_INSURANCE_NUMBER]\nCredit Card Number: ***************3456"
    )


@pytest.mark.timeout(300)
def test_pii_management_in_calm_bot_deletion(test_kafka_consumer: Consumer) -> None:
    """Test the deletion of PII data in the tracker store."""
    sender_id = send_user_messages_to_rasa_pro(HTTP_RASA_SERVER_DELETE)

    # test that the initial tracker contains the expected PII data
    initial_tracker = retrieve_tracker_for_sender_id(HTTP_RASA_SERVER_DELETE, sender_id)

    initial_tracker_events = initial_tracker.get("events", [])
    initial_user_events = list(
        filter(lambda m: m["event"] == "user", initial_tracker_events)
    )
    initial_slot_events = list(
        filter(lambda m: m["event"] == "slot", initial_tracker_events)
    )
    initial_bot_events = list(
        filter(lambda m: m["event"] == "bot", initial_tracker_events)
    )

    assert len(initial_user_events) > 0
    assert len(initial_slot_events) > 0
    assert len(initial_bot_events) > 0

    assert initial_user_events[1]["text"] == "My name is John Doe."
    assert (
        initial_user_events[2]["text"] == "My national insurance number is AB123456C."
    )
    assert (
        initial_user_events[3]["text"]
        == "My credit card number is 1234-5678-9012-3456."
    )
    assert (
        initial_user_events[4]["text"] == "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account "
        "number 2715500356."
    )

    assert initial_slot_events[2]["name"] == "full_name"
    assert initial_slot_events[2]["value"] == "John Doe"
    assert initial_slot_events[3]["name"] == "full_name"
    assert initial_slot_events[3]["value"] == "JOHN DOE"
    assert initial_slot_events[4]["name"] == "national_insurance_number"
    assert initial_slot_events[4]["value"] == "AB123456C"
    assert initial_slot_events[5]["name"] == "credit_card_number"
    assert initial_slot_events[5]["value"] == "1234-5678-9012-3456"

    assert initial_bot_events[4]["text"] == (
        "Your payment has been submitted successfully with the following details:\n\n"
        "Full Name: JOHN DOE\nNational Insurance Number: AB123456C\n"
        "Credit Card Number: 1234-5678-9012-3456"
    )

    time.sleep(120)  # wait for deletion job to complete

    tracker = retrieve_tracker_for_sender_id(HTTP_RASA_SERVER_DELETE, sender_id)

    # we expect the initial tracker to be deleted,
    # so this tracker should be newly created by the MessageProcessor,
    # and it should not contain any of the former events
    # instead it should only contain the events related to the
    # start of the session
    tracker_events = tracker.get("events", [])
    user_events = list(filter(lambda m: m["event"] == "user", tracker_events))
    slot_events = list(filter(lambda m: m["event"] == "slot", user_events))
    bot_events = list(filter(lambda m: m["event"] == "bot", tracker_events))
    assert len(user_events) == 0, "User events should be empty after deletion"
    assert len(slot_events) == 0, "Slot events should be empty after deletion"
    assert len(bot_events) == 0, "Bot events should be empty after deletion"
