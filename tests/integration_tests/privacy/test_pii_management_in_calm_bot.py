import json
import os
import threading
import time
from typing import Any, Dict, Generator, List, Optional

import pytest
import requests
import sqlalchemy as sa
from confluent_kafka import Consumer
from sqlalchemy import URL

from tests.integration_tests.conftest import send_message_to_rasa_server

HTTP_RASA_SERVER_ANONYMIZE = "http://localhost:5005"
HTTP_RASA_SERVER_DELETE = "http://localhost:5006"
# New trackers without env-set (event-only path), session 1 min expiry
HTTP_RASA_SERVER_DELETE_NO_ENV_EXPIRY_FALSE = "http://localhost:5007"
HTTP_RASA_SERVER_DELETE_NO_ENV_EXPIRY_TRUE = "http://localhost:5008"
HTTP_RASA_SERVER_ANONYMIZE_NO_ENV_EXPIRY_FALSE = "http://localhost:5009"
HTTP_RASA_SERVER_ANONYMIZE_NO_ENV_EXPIRY_TRUE = "http://localhost:5010"

USER_MESSAGES = [
    "I want to make a tax payment.",
    "My name is John Doe.",
    "My national insurance number is AB123456C.",
    "My credit card number is 1234-5678-9012-3456.",
    "I've had a great experience, thank you! "
    "Shame that you are not able to process my direct debit "
    "payments from my bank account number 2715500356.",
    "/restart",
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
    {-1: "Is there anything else I can help you with?"},
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


def send_user_messages_to_rasa_pro(
    server_location: str,
    sender_id: Optional[str] = None,
    user_messages: Optional[List[str]] = None,
    bot_messages: Optional[List[Dict[int, str]]] = None,
) -> str:
    user_messages = user_messages or USER_MESSAGES
    bot_messages = bot_messages or BOT_MESSAGES

    for user_message, bot_message in zip(user_messages, bot_messages):
        sender_id, response_messages = send_message_to_rasa_server(
            server_location=server_location, message=user_message, sender_id=sender_id
        )
        for bot_response_index, bot_response_text in bot_message.items():
            assert response_messages, (
                f"Bot returned no messages for {user_message!r}; expected "
                f"{bot_response_text!r}"
            )
            assert (
                response_messages[bot_response_index].get("text") == bot_response_text
            )

    return sender_id


def retrieve_tracker_for_sender_id(
    server_location: str, sender_id: str
) -> Dict[str, Any]:
    tracker_url = (
        f"{server_location}/conversations/{sender_id}/tracker?include_events=ALL"
    )
    response = requests.get(tracker_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Failed to retrieve tracker for sender_id {sender_id}: {response.text}. "
            f"Response code: {response.status_code}"
        )


def _get_tracker_from_db(
    sender_id: str, db_name: Optional[str], start_session_after_expiry: bool
) -> Dict[str, Any]:
    """Load tracker state by querying the SQL tracker store DB directly.

    Expects the standard Rasa events table (id, sender_id, type_name, timestamp,
    intent_name, action_name, data). Returns a dict compatible with API tracker
    response: events, sender_id, terminated.
    """
    if isinstance(db_name, str):
        db_url = URL.create(drivername="sqlite", database=db_name)
    elif start_session_after_expiry:
        db_url = os.getenv("POSTGRES_DB_URL_EXPIRY_TRUE")
    else:
        db_url = os.getenv("POSTGRES_DB_URL_EXPIRY_FALSE")

    engine = sa.create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            sa.text("SELECT data FROM events WHERE sender_id = :sender_id ORDER BY id"),
            {"sender_id": sender_id},
        )
        rows = result.fetchall()
    engine.dispose()
    events = [json.loads(row[0]) for row in rows]
    terminated = bool(events) and events[-1].get("event") == "session_ended"
    return {
        "sender_id": sender_id,
        "events": events,
        "terminated": terminated,
    }


def _append_session_ended(server: str, sender_id: str) -> None:
    """Append SessionEnded to the conversation so the tracker is terminated.

    No-env deletion job only runs on terminated trackers (end with SessionEnded).
    """
    url = f"{server}/conversations/{sender_id}/tracker/events"
    response = requests.post(
        url,
        json=[{"event": "session_ended", "timestamp": time.time()}],
        headers={"Content-Type": "application/json"},
    )
    assert (
        response.status_code == 200
    ), f"Failed to append session_ended: {response.status_code} {response.text}"


def _run_no_env_expiry_flow(
    server: str,
    start_session_after_expiry: bool,
    follow_up_user_messages: List[str],
    follow_up_bot_messages: List[Dict[int, str]],
    wait_after_follow_up_seconds: int = 150,
    terminate_session_before_wait: bool = False,
    db_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full PII flow, wait 65s expiry, send follow-up, wait for cron.

    The SQL tracker store is queried directly for final tracker state.
    """
    sender_id = send_user_messages_to_rasa_pro(
        server,
        user_messages=USER_MESSAGES[:-1],
        bot_messages=BOT_MESSAGES[:-1],
    )
    initial = retrieve_tracker_for_sender_id(server, sender_id)
    initial_user = [e for e in initial.get("events", []) if e.get("event") == "user"]
    assert len(initial_user) >= 5
    if follow_up_user_messages:
        time.sleep(65)
        send_user_messages_to_rasa_pro(
            server,
            sender_id,
            user_messages=follow_up_user_messages,
            bot_messages=follow_up_bot_messages,
        )
    if terminate_session_before_wait:
        _append_session_ended(server, sender_id)
        updated = retrieve_tracker_for_sender_id(server, sender_id)
        assert (
            updated.get("terminated") is True
        ), "Tracker should be terminated before waiting for cron"

    time.sleep(wait_after_follow_up_seconds)

    if terminate_session_before_wait:
        return _get_tracker_from_db(sender_id, db_name, start_session_after_expiry)
    else:
        return retrieve_tracker_for_sender_id(server, sender_id)


def _assert_deletion_no_env_tracker(
    tracker: Dict[str, Any],
) -> None:
    """Assert only the follow-up user message is retained after deletion."""
    events = tracker.get("events", [])
    assert len(events) == 0


def _assert_anonymization_no_env_tracker(tracker: Dict[str, Any]) -> None:
    """Assert anonymized PII ([FULL_NAME])."""
    events = tracker.get("events", [])
    user_events = [e for e in events if e.get("event") == "user"]
    assert len(user_events) == 7
    assert any("[FULL_NAME]" in (e.get("text") or "") for e in user_events)

    slot_events = [e for e in events if e.get("event") == "slot"]
    assert any(
        e.get("value") == "[FULL_NAME]"
        for e in slot_events
        if e.get("name") == "full_name"
    ), "Expected anonymized full_name slot with value [FULL_NAME]"


def _send_single_message_with_retry(
    server: str,
    sender_id: str,
    message: str,
    max_attempts: int = 4,
    retry_delay_seconds: float = 2.0,
) -> None:
    """Send one message; retry when bot returns no messages (e.g. lock contention).

    Used in race tests where cron and user messages contend for the same lock.
    """
    last_error: Optional[AssertionError] = None
    for attempt in range(max_attempts):
        _, response_messages = send_message_to_rasa_server(
            server_location=server, message=message, sender_id=sender_id
        )
        if response_messages:
            return
        last_error = AssertionError(
            f"Expected bot response for message {message!r}\nassert []"
        )
        if attempt < max_attempts - 1:
            time.sleep(retry_delay_seconds)
    raise last_error


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
        "number [BANK ACCOUNT NUMBER]."
    )
    assert (
        user_messages[4]["value"]["parse_data"]["text"]
        == "I've had a great experience, thank you! "
        "Shame that you are not able to process "
        "my direct debit payments from my bank "
        "account number [BANK ACCOUNT NUMBER]."
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
    assert slot_events[6]["value"]["name"] == "feedback"
    assert slot_events[6]["value"]["value"] == (
        "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account number [BANK ACCOUNT NUMBER]."
    )

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

    # bank account number is an entity detected by gliner model
    assert (
        tracker_user_events[4]["text"] == "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account number [BANK ACCOUNT NUMBER]."
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
    assert tracker_slot_events[6]["name"] == "feedback"
    assert tracker_slot_events[6]["value"] == (
        "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account number [BANK ACCOUNT NUMBER]."
    )

    tracker_bot_events = list(filter(lambda m: m["event"] == "bot", tracker_events))
    assert (
        tracker_bot_events[4]["text"]
        == "Your payment has been submitted successfully with the following "
        "details:\n\nFull Name: [FULL_NAME]\nNational Insurance Number: "
        "[NATIONAL_INSURANCE_NUMBER]\nCredit Card Number: ***************3456"
    )


@pytest.mark.timeout(300)
def test_pii_management_in_calm_bot_deletion() -> None:
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
    slot_events = list(
        filter(
            lambda m: m["event"] == "slot"
            and m["name"]
            in {
                "full_name",
                "feedback",
                "national_insurance_number",
                "credit_card_number",
            },
            tracker_events,
        )
    )
    bot_events = list(filter(lambda m: m["event"] == "bot", tracker_events))
    assert len(user_events) == 0, "User events should be empty after deletion"
    assert len(slot_events) == 0, "Slot events should be empty after deletion"
    assert len(bot_events) == 0, "Bot events should be empty after deletion"


@pytest.mark.timeout(300)
def test_pii_management_in_calm_bot_deletion_multiple_tracker_sessions() -> None:
    """Test the deletion of PII data when the tracker contains multiple sessions."""
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

    # wait to add another session to the tracker
    time.sleep(60)
    send_user_messages_to_rasa_pro(
        HTTP_RASA_SERVER_DELETE,
        sender_id,
        user_messages=[
            "I want to make a new tax payment.",
            "Just to confirm, my name is still John Doe.",
        ],
        bot_messages=[
            {-1: "What is your full name?"},
            {-1: "What is your National Insurance Number?"},
        ],
    )

    time.sleep(60)  # wait for deletion job to complete

    tracker = retrieve_tracker_for_sender_id(HTTP_RASA_SERVER_DELETE, sender_id)

    # we expect only the first session to be deleted
    tracker_events = tracker.get("events", [])
    user_events = list(filter(lambda m: m["event"] == "user", tracker_events))
    slot_events = list(filter(lambda m: m["event"] == "slot", tracker_events))
    bot_events = list(filter(lambda m: m["event"] == "bot", tracker_events))

    assert len(user_events) == 2
    assert len(slot_events) > 0
    assert len(bot_events) == 3

    assert user_events[0]["text"] == "I want to make a new tax payment."
    assert user_events[1]["text"] == "Just to confirm, my name is still John Doe."

    assert slot_events[2]["name"] == "full_name"
    assert slot_events[2]["value"] == "John Doe"
    assert slot_events[3]["name"] == "full_name"
    assert slot_events[3]["value"] == "JOHN DOE"

    assert bot_events[1]["text"] == "What is your full name?"
    assert bot_events[2]["text"] == "What is your National Insurance Number?"


@pytest.mark.timeout(300)
def test_pii_management_deletion_no_env_start_session_after_expiry_true() -> None:
    """No-env deletion only runs on terminated trackers (SessionEnded).

    Run flow, expire, send follow-up, then end session (SessionEnded). Cron
    runs on terminated tracker and drops all sessions.
    """
    tracker = _run_no_env_expiry_flow(
        HTTP_RASA_SERVER_DELETE_NO_ENV_EXPIRY_TRUE,
        True,
        ["I want to make another payment."],
        [{-1: "What is your full name?"}],
        wait_after_follow_up_seconds=70,
        terminate_session_before_wait=True,
    )
    _assert_deletion_no_env_tracker(tracker)


@pytest.mark.timeout(300)
def test_pii_management_deletion_no_env_start_session_after_expiry_false() -> None:
    """No-env deletion only runs on terminated trackers (SessionEnded).

    Run flow, expire, send follow-up, then end session (SessionEnded). Cron
    runs on terminated tracker and drops the entire tracker.
    """
    tracker = _run_no_env_expiry_flow(
        HTTP_RASA_SERVER_DELETE_NO_ENV_EXPIRY_FALSE,
        False,
        ["I want to make another payment."],
        [{-1: "What is your full name?"}],
        wait_after_follow_up_seconds=70,
        terminate_session_before_wait=True,
    )
    _assert_deletion_no_env_tracker(tracker)


# Re-engage bot after expiry, then send PII
# so we get a response and retain this PII in tracker
_FOLLOW_UP_WITH_PII = [
    "I want to make another payment.",
    "My name is still John Doe.",
]
_FOLLOW_UP_BOT_PII = [
    {-1: "What is your full name?"},
    {-1: "What is your National Insurance Number?"},
]


@pytest.mark.timeout(300)
def test_pii_management_anonymization_no_env_start_session_after_expiry_true() -> None:
    """New trackers without env-set: session 1min, start_session_after_expiry True.

    After expiry the next message starts a new session (event-only path).
    First session gets ConversationInactive; after min_after_session_end (2 min)
    anonymization runs and anonymizes the first session. Second session may also
    be anonymized after 2 min; we assert the last user message is retained and
    at least some PII was anonymized (e.g. [FULL_NAME] in slots or user events).
    """
    tracker = _run_no_env_expiry_flow(
        HTTP_RASA_SERVER_ANONYMIZE_NO_ENV_EXPIRY_TRUE,
        True,
        _FOLLOW_UP_WITH_PII,
        _FOLLOW_UP_BOT_PII,
        wait_after_follow_up_seconds=200,
    )
    _assert_anonymization_no_env_tracker(tracker)


@pytest.mark.timeout(300)
def test_pii_management_anonymization_no_env_start_session_after_expiry_false() -> None:
    """New trackers without env-set: session 1min, start_session_after_expiry False.

    After expiry the next message does not start a new session (same run).
    """
    tracker = _run_no_env_expiry_flow(
        HTTP_RASA_SERVER_ANONYMIZE_NO_ENV_EXPIRY_FALSE,
        False,
        _FOLLOW_UP_WITH_PII,
        _FOLLOW_UP_BOT_PII,
        wait_after_follow_up_seconds=200,
    )
    _assert_anonymization_no_env_tracker(tracker)


# --- Race condition tests (cron job vs user sending messages) ---


@pytest.mark.timeout(300)
def test_pii_management_race_cron_locking_user_sends() -> None:
    """Cron holds lock while user sends: no lost messages, PII handled by cron.

    Run full PII flow, wait until just before cron (e.g. 119s). Send two
    follow-up messages with PII in quick succession so they contend with the
    cron job. Assert both messages are in the tracker and anonymization has run
    (PII in race messages may be retained or anonymized depending on lock order).
    """
    server = HTTP_RASA_SERVER_ANONYMIZE
    sender_id = send_user_messages_to_rasa_pro(
        server,
        user_messages=USER_MESSAGES[0:2],
        bot_messages=BOT_MESSAGES[0:2],
    )
    initial = retrieve_tracker_for_sender_id(server, sender_id)
    initial_user = [e for e in initial.get("events", []) if e.get("event") == "user"]
    assert len(initial_user) == 2

    # wait 2 extra minutes for the first segment to be anonymized by cron
    now = time.time()
    next_minute = (int(now / 60) + 2) * 60

    # start all threads just after 1 minute past the hour
    # to coincide with cron job timing (cron every 1 min)
    time_to_wait = (
        next_minute - now + 1
    )  # start 1s after the minute to ensure contention
    time.sleep(time_to_wait)

    errors: List[Exception] = []

    def send_at(msg: str, delay: float) -> None:
        time.sleep(delay)
        try:
            _send_single_message_with_retry(server, sender_id, msg)
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=send_at, args=(USER_MESSAGES[2], 1.0))
    t2 = threading.Thread(target=send_at, args=(USER_MESSAGES[3], 1.5))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors, f"Threads raised: {errors}"

    # Allow cron to finish and state to settle
    time.sleep(15)
    tracker = retrieve_tracker_for_sender_id(server, sender_id)
    user_events = [e for e in tracker.get("events", []) if e.get("event") == "user"]
    assert len(user_events) == 4
    texts = [e.get("text", "") for e in user_events]

    # Anonymization job ran on eligible (ended) segment only
    events = tracker.get("events", [])
    slot_events = [e for e in events if e.get("event") == "slot"]
    has_anonymized = any(
        e.get("value") == "[FULL_NAME]"
        for e in slot_events
        if e.get("name") == "full_name"
    ) or any("[FULL_NAME]" in (e.get("text") or "") for e in user_events)
    assert has_anonymized, "Expected anonymized PII ([FULL_NAME]) in tracker after cron"

    first_2_msgs = texts[0:2]
    assert first_2_msgs == [
        "I want to make a tax payment.",
        "My name is [FULL_NAME].",
    ]

    # The later two messages should not be anonymized by cron
    # since the segment was not eligible
    assert USER_MESSAGES[2] in texts
    assert USER_MESSAGES[3] in texts


@pytest.mark.timeout(300)
def test_pii_management_race_cron_during_user_messages() -> None:
    """Cron runs while user is still sending messages: no lost events, PII handled.

    Start partial flow (3 messages), wait for session expiry (65s), send
    follow-ups with PII, then send two more with PII concurrently to race with
    cron. Assert all user messages are present and PII is either retained or
    anonymized by cron.
    """
    server = HTTP_RASA_SERVER_ANONYMIZE
    # First 2 messages of the flow
    sender_id = send_user_messages_to_rasa_pro(
        server,
        user_messages=USER_MESSAGES[0:2],
        bot_messages=BOT_MESSAGES[0:2],
    )
    initial = retrieve_tracker_for_sender_id(server, sender_id)
    initial_user = [e for e in initial.get("events", []) if e.get("event") == "user"]
    assert len(initial_user) == 2

    # Wait for cron to run (every 1 min) and session to be eligible
    # for anonymization
    now = time.time()
    next_minute = (int(now / 60) + 2) * 60
    time_to_wait = next_minute - now  # start 1s before the minute to ensure contention
    time.sleep(time_to_wait)

    # Then send three more with PII concurrently
    errors_a: List[Exception] = []
    errors_b: List[Exception] = []
    errors_c: List[Exception] = []

    def send_a() -> None:
        try:
            _send_single_message_with_retry(server, sender_id, USER_MESSAGES[2])
        except Exception as e:
            errors_a.append(e)

    def send_b() -> None:
        try:
            _send_single_message_with_retry(server, sender_id, USER_MESSAGES[3])
        except Exception as e:
            errors_b.append(e)

    def send_c() -> None:
        try:
            _send_single_message_with_retry(server, sender_id, USER_MESSAGES[4])
        except Exception as e:
            errors_c.append(e)

    ta = threading.Thread(target=send_a)
    tb = threading.Thread(target=send_b)
    tc = threading.Thread(target=send_c)

    # start all threads at 1 minute past the hour
    # to coincide with cron job timing (cron every 1 min)
    now = time.time()
    next_minute = (int(now / 60) + 1) * 60
    time_to_wait = max(
        0, next_minute - now - 1
    )  # start ~1s before the minute to ensure contention
    time.sleep(time_to_wait)
    ta.start()
    tb.start()
    tc.start()

    ta.join()
    tb.join()
    tc.join()
    assert not errors_a, f"Follow-up A raised: {errors_a}"
    assert not errors_b, f"Follow-up B raised: {errors_b}"
    assert not errors_c, f"Follow-up C raised: {errors_c}"

    time.sleep(15)
    tracker = retrieve_tracker_for_sender_id(server, sender_id)

    user_events = [e for e in tracker.get("events", []) if e.get("event") == "user"]
    texts = [e.get("text", "") for e in user_events]

    assert (
        len(user_events) == 5
    ), f"Expected 5 user events, got {len(user_events)}: {texts}"
    first_2_msgs = texts[0:2]
    assert first_2_msgs == [
        "I want to make a tax payment.",
        "My name is [FULL_NAME].",
    ]

    # No duplicate user events (each follow-up once)
    assert sum(1 for t in texts if USER_MESSAGES[2] in t) == 1
    assert sum(1 for t in texts if USER_MESSAGES[3] in t) == 1
    assert sum(1 for t in texts if USER_MESSAGES[4] in t) == 1

    # Active session: follow-up messages must retain PII (cron must not anonymize them)
    last_3_msgs = texts[-3:]
    assert USER_MESSAGES[2] in last_3_msgs
    assert USER_MESSAGES[3] in last_3_msgs
    assert USER_MESSAGES[4] in last_3_msgs
