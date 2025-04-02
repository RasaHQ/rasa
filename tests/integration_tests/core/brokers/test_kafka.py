import logging.config

import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from rasa.core.brokers.kafka import KafkaEventBroker


@pytest.mark.broker
async def test_kafka_event_broker_valid():
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )

    try:
        broker.publish(
            {"sender_id": "valid_test", "event": "user", "text": "hello world!"},
            retries=5,
        )
        assert broker.producer.poll() == 1
    finally:
        await broker.close()


@pytest.mark.broker
async def test_kafka_event_broker_buffer_error_is_handled(caplog: LogCaptureFixture):
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
        queue_size=1,
    )

    event_count = 100
    try:
        for i in range(event_count):
            with caplog.at_level(logging.DEBUG):
                broker.publish(
                    {
                        "sender_id": "valid_test",
                        "event": "user",
                        "text": "hello world!",
                    },
                    retries=5,
                )
        assert "Queue full" in caplog.text
        assert broker.producer.poll() == 1
    finally:
        await broker.close()


@pytest.mark.broker
async def test_kafka_event_broker_handles_message_size_is_too_large(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    """Test that the KafkaEventBroker sends error messages when the original event is too large."""  # noqa: E501
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )
    mb_string = "ab" * 1024 * 1024  # over 1MB
    event = {
        "sender_id": "message_size_test",
        "event": "user",
        "text": mb_string,
    }

    try:
        with caplog.at_level(logging.WARNING):
            broker.publish(event, retries=2, retry_delay_in_seconds=1)
        assert "Message size is too large for the Kafka broker." in caplog.text
        assert broker.producer.poll() == 1
    finally:
        await broker.close()
