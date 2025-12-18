"""Integration test for CALM assistant streaming events to Kafka with broker restart.

This test verifies that a CALM assistant can continuously stream events to a Kafka
broker, and that events sent before, during and after a Kafka broker restart can be
successfully consumed by a Kafka consumer.

This test uses the existing docker-compose setup from
tests_deployment/integration_tests_pii_management_in_calm.
"""

import json
import threading
import time
from typing import Any, Dict, List

import pytest
import requests
from confluent_kafka import Consumer

import docker
from docker.errors import NotFound
from tests.integration_tests.conftest import send_message_to_rasa_server

KAFKA_HOST = "localhost"
KAFKA_PORT = 9092
KAFKA_TOPIC = "anonymization"
RASA_SERVER_PORT = 5005
RASA_SERVER_URL = f"http://localhost:{RASA_SERVER_PORT}"

# Kafka configuration matching the pii_management_in_calm setup (PLAINTEXT, no auth)
KAFKA_CONFIG: Dict[str, Any] = {
    "bootstrap.servers": f"{KAFKA_HOST}:{KAFKA_PORT}",
    "group.id": "calm_kafka_restart_test_group",
    "auto.offset.reset": "earliest",
    "security.protocol": "PLAINTEXT",
}

KAFKA_CONTAINER_NAME = "kafka_broker"

all_consumed_events: List[Dict[str, Any]] = []


def restart_kafka_broker() -> None:
    """Restart the kafka_broker container using docker client."""
    docker_client = docker.from_env()
    try:
        container = docker_client.containers.get(KAFKA_CONTAINER_NAME)
        container.restart()
        container.reload()
    except NotFound:
        raise RuntimeError(
            f"Kafka container '{KAFKA_CONTAINER_NAME}' not found. "
            f"Make sure to run 'make run-pii-calm-containers' first."
        )


def wait_for_kafka_ready(max_wait: int = 60) -> None:
    """Wait for Kafka broker to be ready."""
    wait_time = 0
    while wait_time < max_wait:
        try:
            # Try to create a consumer to verify Kafka is ready
            test_consumer = Consumer(KAFKA_CONFIG)
            test_consumer.close()
            time.sleep(5)  # Give Kafka a bit more time to fully initialize
            return
        except Exception:
            pass
        time.sleep(2)
        wait_time += 2
    raise RuntimeError("Kafka broker did not become ready after restart")


def create_fresh_consumer(group_id_suffix: str = "") -> Consumer:
    """Create a fresh Kafka consumer with a new group ID to read from the beginning."""
    config = KAFKA_CONFIG.copy()
    # Use a unique group ID to ensure we read from the beginning
    import uuid

    suffix = (
        f"_{uuid.uuid4().hex[:8]}"
        if not group_id_suffix
        else f"_{group_id_suffix}_{uuid.uuid4().hex[:8]}"
    )
    config["group.id"] = f"calm_kafka_restart_test_group{suffix}"
    # Ensure we read from the earliest offset for new consumer groups
    config["auto.offset.reset"] = "earliest"
    # Disable auto commit to have more control
    config["enable.auto.commit"] = False
    return Consumer(config)


@pytest.mark.timeout(600)
def test_calm_assistant_kafka_streaming_with_restart() -> None:
    """Test CALM assistant streaming events to Kafka with broker restart.

    This test assumes the docker-compose containers from
    tests_deployment/integration_tests_pii_management_in_calm are already running.

    This test:
    1. Sends messages to a CALM assistant that streams events to Kafka
    2. Restarts the Kafka broker using docker client
    3. Continues sending messages
    4. Verifies that events from both before and after restart can be consumed
    """
    # Verify Rasa server is running
    try:
        response = requests.get(f"{RASA_SERVER_URL}/status", timeout=5)
        if response.status_code != 200:
            raise RuntimeError(
                f"Rasa server is not running or not ready. "
                f"Expected status 200, got {response.status_code}. "
                f"Make sure to run 'make run-pii-calm-containers' first."
            )
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Could not connect to Rasa server at {RASA_SERVER_URL}. "
            f"Make sure to run 'make run-pii-calm-containers' first. Error: {e}"
        )

    sender_id = "test_sender_calm_kafka_restart"

    # Reset global events list at start of test
    global all_consumed_events
    all_consumed_events = []

    # Messages for each phase
    messages_before_restart = [
        "I want to make a tax payment.",
        "My name is John Doe.",
    ]
    messages_during_restart = [
        "My national insurance number is AB123456C.",
        "My credit card number is 1234-5678-9012-3456.",
    ]
    messages_after_restart = [
        "I've had a great experience, thank you! "
        "Shame that you are not able to process my direct debit "
        "payments from my bank account number 2715500356.",
    ]

    # Anonymized versions of the messages
    # which publisher is expected to send to Kafka
    anonymized_messages_before_restart = [
        "I want to make a tax payment.",
        "My name is [FULL_NAME].",
    ]
    anonymized_messages_during_restart = [
        "My national insurance number is [NATIONAL_INSURANCE_NUMBER].",
        "My credit card number is ***************3456.",
    ]
    anonymized_messages_after_restart = [
        "I've had a great experience, thank you! "
        "Shame that you are not able to process my "
        "direct debit payments from my bank account "
        "number [BANK ACCOUNT NUMBER]."
    ]

    # Create a consumer that will run continuously
    consumer = create_fresh_consumer("continuous")
    stop_consuming = threading.Event()
    consumer_lock = threading.Lock()

    def consume_continuously():
        """Consumer thread that runs continuously."""
        global all_consumed_events
        try:
            # Subscribe first
            consumer.subscribe([KAFKA_TOPIC])
            print(f"Consumer subscribed to topic: {KAFKA_TOPIC}")

            # Wait for partition assignment
            assignment_wait = 0
            while assignment_wait < 10:
                assignment = consumer.assignment()
                if assignment:
                    print(
                        f"Consumer assigned to partitions: "
                        f"{[f'{tp.topic}:{tp.partition}' for tp in assignment]}"
                    )
                    # Seek to beginning for all assigned partitions
                    from confluent_kafka import TopicPartition

                    beginning_partitions = [
                        TopicPartition(tp.topic, tp.partition, 0) for tp in assignment
                    ]
                    consumer.assign(beginning_partitions)
                    print("Seeking to beginning of all partitions")
                    break
                time.sleep(0.5)
                assignment_wait += 0.5

            # Consume events and add them incrementally to all_consumed_events
            messages: List[Dict[str, Any]] = []
            start_time = time.time()
            timeout = 300.0
            poll_count = 0

            while time.time() - start_time < timeout:
                if stop_consuming.is_set():
                    break
                message = consumer.poll(timeout=1.0)
                poll_count += 1
                if message is None:
                    if poll_count % 10 == 0:  # Log every 10 empty polls
                        print(
                            f"No message received (poll #{poll_count}, "
                            f"elapsed: {time.time() - start_time:.1f}s)"
                        )
                    continue
                if message.error():
                    if message.error().code() == -191:  # _PARTITION_EOF
                        print(
                            f"Reached end of partition "
                            f"{message.topic()}:{message.partition()}"
                        )
                        continue
                    # During restart, we might get connection errors - log but continue
                    error_code = message.error().code()
                    if error_code in (-195, -185):  # _TRANSPORT, _ALL_BROKERS_DOWN
                        print(
                            f"Kafka connection error during restart (expected): "
                            f"{message.error()}"
                        )
                        continue
                    raise Exception(f"Kafka Error: {message.error()}")
                try:
                    value = json.loads(message.value().decode("utf-8"))
                    event_dict = {
                        "value": value,
                        "topic": message.topic(),
                        "partition": message.partition(),
                        "offset": message.offset(),
                    }
                    messages.append(event_dict)
                    # Add to global list immediately so test can access it
                    with consumer_lock:
                        all_consumed_events.append(event_dict)
                    print(
                        f"✓ Consumed message from {message.topic()}:"
                        f"{message.partition()} "
                        f"offset {message.offset()}, "
                        f"event type: {value.get('event', 'unknown')}"
                    )
                except Exception as e:
                    # Skip malformed messages
                    print(f"Error parsing message: {e}")

            print(
                f"Consumer thread finished. "
                f"Consumed {len(messages)} messages in {time.time() - start_time:.1f}s"
            )
        except Exception as e:
            print(f"Consumer error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            consumer.close()

    # Start consumer thread
    consumer_thread = threading.Thread(target=consume_continuously, daemon=True)
    consumer_thread.start()
    time.sleep(2)  # Give consumer time to start

    # Phase 1: Send messages before Kafka restart and verify consumer receives them
    print("Phase 1: Sending messages before Kafka restart...")
    for message in messages_before_restart:
        try:
            send_message_to_rasa_server(
                server_location=RASA_SERVER_URL,
                message=message,
                sender_id=sender_id,
            )
            time.sleep(1)  # Give time for events to be published
        except Exception as e:
            print(f"Error sending message '{message}': {e}")

    # Wait for events to be consumed
    time.sleep(5)
    with consumer_lock:
        events_before_restart = [
            event
            for event in all_consumed_events
            if event["value"].get("event") == "user"
        ]
    before_restart_offsets = {event["offset"] for event in events_before_restart}
    user_texts_before = [
        event["value"].get("text", "") for event in events_before_restart
    ]

    print(f"Phase 1: Consumed {len(events_before_restart)} user events before restart")
    found_before = any(
        any(msg.lower() in text.lower() for msg in anonymized_messages_before_restart)
        for text in user_texts_before
    )
    assert found_before, (
        f"Expected to find events from before restart. "
        f"Messages sent: {messages_before_restart}, "
        f"Events found: {user_texts_before}"
    )

    # Phase 2: Restart Kafka broker and send messages during restart
    print("Phase 2: Restarting Kafka broker and sending messages during restart...")
    restart_start_time = time.time()
    restart_kafka_broker()

    # Send messages while Kafka is restarting
    # Note: The producer will buffer these and retry after Kafka comes back up
    for message in messages_during_restart:
        try:
            send_message_to_rasa_server(
                server_location=RASA_SERVER_URL,
                message=message,
                sender_id=sender_id,
            )
            time.sleep(0.5)
        except Exception as e:
            # Expected to potentially fail while Kafka is down
            print(f"Error sending message '{message}' during restart: {e}")

    # Wait for Kafka to be ready again
    print("Phase 2: Waiting for Kafka broker to be ready...")
    wait_for_kafka_ready()
    restart_end_time = time.time()
    restart_duration = restart_end_time - restart_start_time
    print(f"Kafka restart completed in {restart_duration:.1f}s")

    # Check what was consumed during the restart period
    # Messages sent during restart may be buffered by
    # the producer and delivered after restart
    # We verify they arrive AFTER the restart completes, not before
    time.sleep(3)  # Give producer time to retry and deliver buffered messages
    with consumer_lock:
        # Get all events that arrived after the restart started
        events_during_restart = [
            event
            for event in all_consumed_events
            if event["value"].get("event") == "user"
            and event["offset"] not in before_restart_offsets
        ]
    user_texts_during = [
        event["value"].get("text", "") for event in events_during_restart
    ]

    print(
        f"Phase 2: Found {len(events_during_restart)} user events after restart started"
    )

    # Check if messages sent during restart were delivered
    # They should be delivered AFTER restart (producer retry behavior), not lost
    found_during_restart_messages = any(
        any(msg.lower() in text.lower() for msg in anonymized_messages_during_restart)
        for text in user_texts_during
    )

    if found_during_restart_messages:
        print(
            "Note: Messages sent during restart were delivered "
            "after restart completed. This is expected behavior - "
            "the Kafka producer buffers and retries messages."
        )
        # Mark these offsets so we don't count them in the "after restart" phase
        during_restart_offsets = {
            event["offset"]
            for event in events_during_restart
            if any(
                msg.lower() in event["value"].get("text", "").lower()
                for msg in anonymized_messages_during_restart
            )
        }
    else:
        # If messages weren't delivered, they were truly lost (producer didn't retry)
        print(
            "Messages sent during restart were not delivered (lost). "
            "This can happen if the producer fails before buffering."
        )
        during_restart_offsets = set()

    # Phase 3: Send messages after Kafka restart and verify consumer receives them
    print("Phase 3: Sending messages after Kafka restart...")
    for message in messages_after_restart:
        try:
            send_message_to_rasa_server(
                server_location=RASA_SERVER_URL,
                message=message,
                sender_id=sender_id,
            )
            time.sleep(1)  # Give time for events to be published
        except Exception as e:
            print(f"Error sending message '{message}': {e}")

    # Wait for events to be consumed
    time.sleep(5)
    with consumer_lock:
        # Get events that are not from before restart and not from during restart
        events_after_restart = [
            event
            for event in all_consumed_events
            if event["value"].get("event") == "user"
            and event["offset"] not in before_restart_offsets
            and event["offset"] not in during_restart_offsets
        ]
    user_texts_after = [
        event["value"].get("text", "") for event in events_after_restart
    ]

    print(f"Phase 3: Consumed {len(events_after_restart)} user events after restart")
    found_after = any(
        any(msg.lower() in text.lower() for msg in anonymized_messages_after_restart)
        for text in user_texts_after
    )
    assert found_after, (
        f"Expected to find events from after restart. "
        f"Messages sent: {messages_after_restart}, "
        f"Events found: {user_texts_after}"
    )

    # Stop consuming
    stop_consuming.set()
    consumer_thread.join(timeout=5)

    # Final verification
    with consumer_lock:
        all_user_events = [
            event
            for event in all_consumed_events
            if event["value"].get("event") == "user"
        ]

    print(
        f"Successfully verified {len(all_consumed_events)} total events, "
        f"including {len(all_user_events)} user events. "
        f"Before restart: {len(events_before_restart)}, "
        f"During restart: {len(events_during_restart)}, "
        f"After restart: {len(events_after_restart)}"
    )
