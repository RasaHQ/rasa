from rasa.core.brokers.kafka import KafkaEventBroker


def test_kafka_event_broker_valid():
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
        broker.producer.flush()
        broker._close()


def test_kafka_event_broker_raises_buffer_error():
    broker = KafkaEventBroker(
        url="localhost",
        topic="rasa",
        sasl_username="admin",
        sasl_password="password",
        partition_by_sender=True,
    )

    event_count = 100000
    for i in range(event_count):
        try:
            broker.publish(
                {"sender_id": "valid_test", "event": "user", "text": "hello world!"},
                retries=5,
            )
            assert broker.producer.poll() == 1
        finally:
            broker.producer.flush()
            broker._close()
