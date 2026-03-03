import logging.config

import pytest
from pytest import LogCaptureFixture, MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.brokers.kafka import KafkaEventBroker
from tests.utilities import filter_logs


@pytest.mark.broker
async def test_kafka_event_broker_valid() -> None:
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )

    try:
        with capture_logs() as caplog:
            broker.publish(
                {"sender_id": "valid_test", "event": "user", "text": "hello world!"},
                retries=5,
            )
            logs = filter_logs(
                caplog,
                "kafka.publish.event",
                "debug",
                [
                    "Logging a reduced version of the Kafka event",
                ],
                log_contains_all_message_parts=False,
            )
            assert len(logs) == 1
            assert logs[0].get("rasa_event") == {
                "sender_id": "valid_test",
                "event": "user",
                "text": "hello world!",
            }
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

    event_count = 2
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
    finally:
        await broker.close()
