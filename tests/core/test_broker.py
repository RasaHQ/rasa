import contextlib
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Type, Union
from unittest.mock import AsyncMock, MagicMock, Mock

import aio_pika.exceptions
import aiormq.exceptions
import confluent_kafka
import pamqp.exceptions
import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from aiormq import ChannelNotFoundEntity
from confluent_kafka import KafkaError, KafkaException, Producer

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.brokers import pika
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.file import FileEventBroker
from rasa.core.brokers.kafka import KafkaEventBroker, KafkaProducerInitializationError
from rasa.core.brokers.pika import DEFAULT_QUEUE_NAME, PikaEventBroker
from rasa.core.brokers.sql import SQLEventBroker
from rasa.core.constants import (
    IAM_CLOUD_PROVIDER_ENV_VAR_NAME,
    KAFKA_MSK_AWS_IAM_ENABLED_ENV_VAR_NAME,
)
from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
    AWSMSKafkaIAMCredentialsProvider,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    TemporaryCredentials,
)
from rasa.shared.core.events import Event, Restarted, SlotSet, UserUttered
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

TEST_EVENTS = [
    UserUttered("/greet", {"name": "greet", "confidence": 1.0}, []),
    SlotSet("name", "rasa"),
    Restarted(),
]


@pytest.mark.asyncio
async def test_pika_broker_from_config(monkeypatch: MonkeyPatch):
    # patch PikaEventBroker so it doesn't try to connect to RabbitMQ on init
    monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock())

    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/pika_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, PikaEventBroker)
    assert actual.host == "localhost"
    assert actual.username == "username"
    assert actual.queues == ["queue-1"]
    assert actual.exchange_name == "exchange"
    assert actual.stream_pii is True
    assert actual.anonymization_queues == []


@pytest.mark.asyncio
async def test_pika_broker_from_config_with_pii(monkeypatch: MonkeyPatch):
    # Mock RabbitMQ connection
    monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock())

    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/pika_with_pii_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, PikaEventBroker)
    assert actual.host == "localhost"
    assert actual.username == "username"
    assert actual.queues == ["queue-1"]
    assert actual.exchange_name == "exchange"
    assert actual.stream_pii is False
    assert actual.anonymization_queues == ["anonymized_queue_1"]


@pytest.mark.asyncio
async def test_pika_message_property_app_id_without_env_set(monkeypatch: MonkeyPatch):
    # unset RASA_ENVIRONMENT env var results in empty App ID
    monkeypatch.delenv("RASA_ENVIRONMENT", raising=False)
    pika_broker = PikaEventBroker("some host", "username", "password")

    assert not pika_broker._message({}, None).app_id


@pytest.mark.asyncio
async def test_pika_message_property_app_id(monkeypatch: MonkeyPatch):
    # setting it to some value results in that value as the App ID
    rasa_environment = "some-test-environment"
    monkeypatch.setenv("RASA_ENVIRONMENT", rasa_environment)
    pika_broker = PikaEventBroker("some host", "username", "password")

    assert pika_broker._message({}, None).app_id == rasa_environment


@pytest.mark.parametrize(
    "queues_arg,expected,warning",
    [
        # default case
        (["q1", "q2"], ["q1", "q2"], None),
        # `queues` arg supplied, as string
        ("q1", ["q1"], None),
        # no queues provided. Use default queue and print warning.
        (None, [DEFAULT_QUEUE_NAME], UserWarning),
    ],
)
@pytest.mark.asyncio
async def test_pika_queues_from_args(
    queues_arg: Union[Text, List[Text], None],
    expected: List[Text],
    warning: Optional[Type[Warning]],
):
    with pytest.warns(warning):
        pika_processor = PikaEventBroker(
            "host",
            "username",
            "password",
            queues=queues_arg,
            get_message=lambda: ("", None),
        )

    assert pika_processor.queues == expected


@pytest.mark.asyncio
async def test_pika_raise_connection_exception(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        PikaEventBroker, "connect", AsyncMock(side_effect=ChannelNotFoundEntity())
    )

    with pytest.raises(ConnectionException):
        await EventBroker.create(
            EndpointConfig(username="username", password="password", type="pika")
        )


@pytest.mark.parametrize(
    "exception",
    (
        RuntimeError,
        ConnectionError,
        OSError,
        aiormq.exceptions.AMQPError,
        pamqp.exceptions.PAMQPException,
        pamqp.exceptions.AMQPConnectionForced,
        pamqp.exceptions.AMQPNotFound,
        pamqp.exceptions.AMQPInternalError,
    ),
)
@pytest.mark.asyncio
async def test_aio_pika_exceptions_caught(
    exception: Exception, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(PikaEventBroker, "connect", AsyncMock(side_effect=exception))

    with pytest.raises(ConnectionException):
        await EventBroker.create(
            EndpointConfig(username="username", password="password", type="pika")
        )


@pytest.mark.asyncio
async def test_no_broker_in_config(endpoints_path: Text):
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await EventBroker.create(cfg)

    assert actual is None


@pytest.mark.asyncio
async def test_sql_broker_from_config():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, SQLEventBroker)
    assert actual.engine.name == "sqlite"


@pytest.mark.asyncio
async def test_sql_broker_logs_to_sql_db():
    cfg = read_endpoint_config(
        "data/test_endpoints/event_brokers/sql_endpoint.yml", "event_broker"
    )
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, SQLEventBroker)

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    with actual.session_scope() as session:
        events_types = [
            json.loads(event.data)["event"]
            for event in session.query(actual.SQLBrokerEvent).all()
        ]

    assert events_types == ["user", "slot", "restart"]


@pytest.mark.asyncio
async def test_file_broker_from_config(tmp_path: Path):
    # backslashes need to be encoded (windows...) otherwise we run into unicode issues
    path = str(tmp_path / "rasa_test_event.log").replace("\\", "\\\\")
    endpoint_config = textwrap.dedent(
        f"""
        event_broker:
          path: "{path}"
          type: "file"
    """
    )
    rasa.shared.utils.io.write_text_file(endpoint_config, tmp_path / "endpoint.yml")

    cfg = read_endpoint_config(str(tmp_path / "endpoint.yml"), "event_broker")
    actual = await EventBroker.create(cfg)

    assert isinstance(actual, FileEventBroker)
    assert actual.path.endswith("rasa_test_event.log")


@pytest.mark.asyncio
async def test_file_broker_logs_to_file(tmp_path: Path):
    log_file_path = str(tmp_path / "events.log")

    actual = await EventBroker.create(
        EndpointConfig(**{"type": "file", "path": log_file_path})
    )

    for e in TEST_EVENTS:
        actual.publish(e.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            recovered.append(Event.from_parameters(json.loads(line)))

    assert recovered == TEST_EVENTS


@pytest.mark.asyncio
async def test_file_broker_properly_logs_newlines(tmp_path: Path):
    log_file_path = str(tmp_path / "events.log")

    actual = await EventBroker.create(
        EndpointConfig(**{"type": "file", "path": log_file_path})
    )

    event_with_newline = UserUttered("hello \n there")

    actual.publish(event_with_newline.as_dict())

    # reading the events from the file one event per line
    recovered = []
    with open(log_file_path, "r") as log_file:
        for line in log_file:
            recovered.append(Event.from_parameters(json.loads(line)))

    assert recovered == [event_with_newline]


@pytest.mark.asyncio
async def test_load_custom_broker_name(tmp_path: Path):
    config = EndpointConfig(
        **{
            "type": "rasa.core.brokers.file.FileEventBroker",
            "path": str(tmp_path / "rasa_event.log"),
        }
    )
    broker = await EventBroker.create(config)
    assert broker


@pytest.mark.asyncio
async def test_load_non_existent_custom_broker_name():
    config = EndpointConfig(**{"type": "rasa.core.brokers.my.MyProducer"})
    assert await EventBroker.create(config) is None


@pytest.mark.asyncio
async def test_kafka_broker_from_config():
    endpoints_path = (
        "data/test_endpoints/event_brokers/kafka_sasl_plaintext_endpoint.yml"
    )
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)

    expected = KafkaEventBroker(
        "localhost",
        sasl_username="username",
        sasl_password="password",
        sasl_mechanism="PLAIN",
        topic="topic",
        partition_by_sender=True,
        security_protocol="SASL_PLAINTEXT",
    )

    assert actual.url == expected.url
    assert actual.sasl_username == expected.sasl_username
    assert actual.sasl_password == expected.sasl_password
    assert actual.sasl_mechanism == expected.sasl_mechanism
    assert actual.topic == expected.topic
    assert actual.partition_by_sender == expected.partition_by_sender
    assert actual.stream_pii is True
    assert actual.anonymization_topics == []


@pytest.mark.parametrize(
    "file,exception",
    [
        ("kafka_sasl_plaintext_endpoint.yml", confluent_kafka.KafkaException),
        ("kafka_plaintext_endpoint.yml", confluent_kafka.KafkaException),
        ("kafka_sasl_ssl_endpoint.yml", KafkaProducerInitializationError),
        ("kafka_ssl_endpoint.yml", KafkaProducerInitializationError),
        # `ValueError` exception is raised when the `security_protocol` is incorrect
        ("kafka_invalid_security_protocol.yml", ValueError),
        # `confluent_kafka.KafkaException` exception is raised when there is no
        # `url` specified
        ("kafka_plaintext_endpoint_no_url.yml", confluent_kafka.KafkaException),
        # `KafkaProducerInitializationError` is raised when an invalid
        # `sasl_mechanism` is provided
        ("kafka_invalid_sasl_mechanism.yml", KafkaProducerInitializationError),
    ],
)
@pytest.mark.asyncio
async def test_kafka_broker_security_protocols(file: Text, exception: Exception):
    endpoints_path = f"data/test_endpoints/event_brokers/{file}"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)
    with pytest.raises(exception):
        # noinspection PyProtectedMember
        producer = actual._create_producer()

        # required action to trigger expected exception because the configuration
        # seems correct and the producer gets instantiated but a connection to the
        # broker cannot be established
        producer.list_topics("topic", timeout=1)


@pytest.mark.flaky
@pytest.mark.asyncio
async def test_no_pika_logs_if_no_debug_mode(caplog: LogCaptureFixture):
    """Tests that when you run rasa with logging set at INFO,
    the debugs from pika dependency are not going to be shown
    """
    broker = PikaEventBroker(
        "host", "username", "password", retry_delay_in_seconds=1, connection_attempts=1
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(Exception):
            await broker.connect()

    # Only Rasa Pro logs, but logs from the library itself.
    assert all(
        record.name
        in [
            "rasa.core.brokers.pika",
            "asyncio",
            "aio_pika.robust_connection",
            "ddtrace.internal.writer.writer",
        ]
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_create_pika_invalid_port():
    cfg = EndpointConfig(
        username="username", password="password", type="pika", port="PORT"
    )
    with pytest.raises(RasaException) as e:
        await EventBroker.create(cfg)
        assert "Port could not be converted to integer." in str(e.value)


@pytest.mark.asyncio
async def test_warning_if_unsupported_ssl_env_variables(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RABBITMQ_SSL_KEY_PASSWORD", "test")
    monkeypatch.setenv("RABBITMQ_SSL_CA_FILE", "test")

    with pytest.warns(UserWarning):
        pika._create_rabbitmq_ssl_options()


@pytest.mark.asyncio
async def test_pika_connection_error(monkeypatch: MonkeyPatch):
    # patch PikaEventBroker to raise an AMQP connection error
    mock_connection = AsyncMock(
        side_effect=aio_pika.exceptions.ProbableAuthenticationError("Oups")
    )
    monkeypatch.setattr(PikaEventBroker, "connect", mock_connection)

    cfg = EndpointConfig.from_dict(
        {
            "type": "pika",
            "url": "localhost",
            "username": "username",
            "password": "password",
            "queues": ["queue-1"],
            "connection_attempts": 1,
            "retry_delay_in_seconds": 0,
        }
    )
    with pytest.raises(ConnectionException):
        await EventBroker.create(cfg)

    mock_connection.assert_called_once()


@pytest.mark.asyncio
async def test_sql_connection_error(monkeypatch: MonkeyPatch):
    cfg = EndpointConfig.from_dict(
        {
            "type": "sql",
            "dialect": "postgresql",
            "url": "0.0.0.0",
            "port": 42,
            "db": "boom",
            "username": "user",
            "password": "pw",
        }
    )
    with pytest.raises(ConnectionException):
        await EventBroker.create(cfg)


@pytest.mark.parametrize(
    "host,expected_url",
    [
        ("localhost", None),
        ("amqp://localhost", "amqp://test_user:test_pass@localhost:5672"),
        (
            "amqp://test_user:test_pass@localhost",
            "amqp://test_user:test_pass@localhost:5672",
        ),
        (
            "amqp://test_user:test_pass@localhost/myvhost?connection_timeout=10",
            "amqp://test_user:test_pass@localhost:5672/myvhost?connection_timeout=10",
        ),
        ("amqp://localhost:5672", "amqp://test_user:test_pass@localhost:5672"),
        (
            "amqp://test_user:test_pass@localhost:5672/myvhost?connection_timeout=10",
            "amqp://test_user:test_pass@localhost:5672/myvhost?connection_timeout=10",
        ),
    ],
)
@pytest.mark.asyncio
async def test_pika_event_broker_configure_url(
    host: Text, expected_url: Optional[Text]
) -> None:
    # deepcode ignore NoHardcodedCredentials/test: Test credential
    username = "test_user"
    # deepcode ignore NoHardcodedPasswords/test: Test credential
    password = "test_pass"
    broker = PikaEventBroker(host=host, username=username, password=password)
    url = broker._configure_url()
    assert url == expected_url


@pytest.mark.asyncio
async def test_kafka_event_broker_handle_message_size_too_large(
    monkeypatch: MonkeyPatch,
) -> None:
    # raise exception only first time when called
    mock_publish = Mock(side_effect=[KafkaException(KafkaError(10)), None])
    mock_retry_connection = Mock()

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer", MagicMock()
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection", Mock()
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._retry_connection",
        mock_retry_connection,
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._publish", mock_publish
    )

    # Given
    event = {
        "sender_id": "message_size_test",
        "event": "user",
        "text": "test",
    }

    # When
    broker = KafkaEventBroker(
        "localhost",
        sasl_username="username",
        sasl_password="password",
        sasl_mechanism="PLAIN",
        topic="topic",
        partition_by_sender=True,
        security_protocol="SASL_PLAINTEXT",
    )
    broker.publish(event, retries=2, retry_delay_in_seconds=1)

    # Then
    assert mock_retry_connection.call_count == 1
    error_event = mock_publish.call_args[0][0]
    assert error_event["event"] == "error"
    assert error_event["error_code"] == 10
    assert error_event["metadata"]["error_source"] == "KafkaEventBroker"
    assert (
        "Skipping message for event type 'user' because of Kafka message size limit"
        in error_event["metadata"]["error_msg"]
    )


@pytest.mark.asyncio
async def test_kafka_broker_from_config_with_pii_attributes():
    endpoints_path = "data/test_endpoints/event_brokers/kafka_pii_endpoint.yml"
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)

    expected = KafkaEventBroker(
        "localhost",
        sasl_username="username",
        sasl_password="password",
        sasl_mechanism="PLAIN",
        topic="topic",
        partition_by_sender=True,
        security_protocol="SASL_PLAINTEXT",
        stream_pii=False,
        anonymization_topics=["anonymized"],
    )

    assert actual.url == expected.url
    assert actual.sasl_username == expected.sasl_username
    assert actual.sasl_password == expected.sasl_password
    assert actual.sasl_mechanism == expected.sasl_mechanism
    assert actual.topic == expected.topic
    assert actual.partition_by_sender == expected.partition_by_sender
    assert actual.stream_pii == expected.stream_pii
    assert actual.anonymization_topics == expected.anonymization_topics


@pytest.mark.asyncio
async def test_kafka_event_broker_iam_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv(KAFKA_MSK_AWS_IAM_ENABLED_ENV_VAR_NAME, "True")
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    broker = KafkaEventBroker(
        "localhost",
        sasl_mechanism="OAUTHBEARER",
        topic="topic",
        partition_by_sender=True,
        security_protocol="SASL_SSL",
        ssl_check_hostname=True,
    )

    assert isinstance(broker.iam_credentials_provider, AWSMSKafkaIAMCredentialsProvider)

    config = broker._get_kafka_config()

    assert config["sasl.mechanism"] == "OAUTHBEARER"
    assert config["security.protocol"] == "SASL_SSL"
    assert config["oauth_cb"] == broker.get_aws_iam_token
    assert "sasl.username" not in config
    assert "sasl.password" not in config


@pytest.fixture
def common_kafka_config() -> Dict[str, Any]:
    return {
        "url": "localhost",
        "topic": "topic",
        "sasl_username": "user",
        "sasl_password": "pass",
        "security_protocol": "SASL_PLAINTEXT",
    }


@pytest.mark.asyncio
async def test_kafka_broker_keepalive_defaults(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """Keepalive attributes have defaults and are not in config when disabled."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    assert broker.socket_keepalive_enable is False
    assert broker.reconnect_backoff_ms == 1000
    assert broker.reconnect_backoff_max_ms == 10000
    assert broker.topic_metadata_refresh_interval_ms == 300000

    config = broker._get_kafka_config()
    assert "socket.keepalive.enable" not in config
    assert "reconnect.backoff.ms" not in config
    assert "reconnect.backoff.max.ms" not in config
    assert "topic.metadata.refresh.interval.ms" not in config


@pytest.mark.asyncio
async def test_kafka_broker_keepalive_config_when_enabled(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """When socket_keepalive_enable is True, producer config includes keepalive settings."""  # noqa: E501
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    kafka_config = {**common_kafka_config, "socket_keepalive_enable": True}
    broker = KafkaEventBroker(
        **kafka_config,
    )
    config = broker._get_kafka_config()
    assert config["socket.keepalive.enable"] is True
    assert config["reconnect.backoff.ms"] == 1000
    assert config["reconnect.backoff.max.ms"] == 10000
    assert config["topic.metadata.refresh.interval.ms"] == 300000


@pytest.mark.asyncio
async def test_kafka_broker_keepalive_custom_values(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """Custom keepalive kwargs are stored and passed into producer config."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    kafka_config = {
        **common_kafka_config,
        "socket_keepalive_enable": True,
        "reconnect_backoff_ms": 500,
        "reconnect_backoff_max_ms": 5000,
        "topic_metadata_refresh_interval_ms": 60000,
    }
    broker = KafkaEventBroker(**kafka_config)
    assert broker.reconnect_backoff_ms == 500
    assert broker.reconnect_backoff_max_ms == 5000
    assert broker.topic_metadata_refresh_interval_ms == 60000

    config = broker._get_kafka_config()
    assert config["socket.keepalive.enable"] is True
    assert config["reconnect.backoff.ms"] == 500
    assert config["reconnect.backoff.max.ms"] == 5000
    assert config["topic.metadata.refresh.interval.ms"] == 60000


@pytest.mark.asyncio
async def test_kafka_broker_invalid_casing_for_sasl_mechanism():
    endpoints_path = (
        "data/test_endpoints/event_brokers/kafka_lower_case_sasl_mechanism.yml"
    )
    cfg = read_endpoint_config(endpoints_path, "event_broker")

    actual = await KafkaEventBroker.from_endpoint_config(cfg)
    with contextlib.nullcontext():
        producer = actual._create_producer()
        assert isinstance(producer, Producer)


@pytest.mark.asyncio
async def test_kafka_poll_loop_calls_producer_poll(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """Background poll loop calls producer.poll(0.1) when producer is set."""
    mock_producer = MagicMock()
    poll_call_count = [0]  # use list so inner function can mutate

    def on_poll(timeout: float = 0.1) -> None:
        poll_call_count[0] += 1
        if poll_call_count[0] == 5:
            broker._cancelled = True

    mock_producer.poll.side_effect = on_poll

    broker = KafkaEventBroker(**common_kafka_config)
    broker.producer = mock_producer

    broker._poll_thread.join(timeout=2.0)

    assert poll_call_count[0] == 5
    assert mock_producer.poll.call_count == 5
    mock_producer.poll.assert_called_with(0.1)


@pytest.mark.asyncio
async def test_kafka_poll_triggers_oauth_cb_for_iam_refresh(
    monkeypatch: MonkeyPatch,
) -> None:
    """When poll runs in background with IAM config, oauth_cb is invoked for token refresh."""  # noqa: E501
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv(KAFKA_MSK_AWS_IAM_ENABLED_ENV_VAR_NAME, "True")
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")

    mock_credentials = TemporaryCredentials(
        auth_token="test-iam-token", expiration=1234567890.0
    )
    mock_provider = MagicMock()
    mock_provider.get_temporary_credentials.return_value = mock_credentials

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.create_iam_credentials_provider",
        lambda _: mock_provider,
    )

    class MockProducerThatCallsOAuthCb:
        """Producer mock that invokes oauth_cb when poll() is called."""

        def __init__(self, config: dict) -> None:
            self._config = config
            self._poll_count = 0

        def poll(self, timeout: float = 0.1) -> int:
            self._poll_count += 1
            oauth_cb = self._config.get("oauth_cb")
            if oauth_cb is not None:
                oauth_cb(self._config.get("oauth_config"))
            if self._poll_count == 5:
                broker._cancelled = True
            return 0

    monkeypatch.setattr(
        "confluent_kafka.Producer",
        MockProducerThatCallsOAuthCb,
    )

    broker = KafkaEventBroker(
        url="localhost",
        topic="topic",
        sasl_mechanism="OAUTHBEARER",
        security_protocol="SASL_SSL",
        ssl_check_hostname=True,
    )
    broker.producer = MockProducerThatCallsOAuthCb(broker._get_kafka_config())

    broker._poll_thread.join(timeout=2.0)

    assert (
        mock_provider.get_temporary_credentials.called
    ), "oauth_cb (get_aws_iam_token) should be invoked on poll and refresh IAM token"
    assert mock_provider.get_temporary_credentials.call_count == 5


@pytest.mark.asyncio
async def test_kafka_publish_initial_connection_failure_returns_early(
    monkeypatch: MonkeyPatch,
    common_kafka_config: Dict[str, Any],
) -> None:
    """publish returns without entering retry loop when initial connection check fails."""  # noqa: E501
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    mock_create = MagicMock(return_value=MagicMock())
    mock_check = Mock(side_effect=KafkaException(KafkaError(1)))
    mock_publish = Mock()

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer", mock_create
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection",
        mock_check,
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._publish", mock_publish
    )

    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    event = {"sender_id": "s1", "event": "user", "text": "hi"}

    broker.publish(event, retries=3)

    mock_create.assert_called_once()
    mock_check.assert_called_once()
    mock_publish.assert_not_called()


@pytest.mark.asyncio
async def test_kafka_publish_queue_full_kafka_exception_polls_and_decrements_retries(
    monkeypatch: MonkeyPatch,
    common_kafka_config: Dict[str, Any],
) -> None:
    """publish handles KafkaException _QUEUE_FULL: polls producer, no _retry_publish."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    mock_producer = MagicMock()
    mock_producer.poll.return_value = 0
    call_count = [0]

    def publish_side_effect(*args: Any, **kwargs: Any) -> None:
        call_count[0] += 1
        if call_count[0] <= 2:
            err = MagicMock()
            err.code.return_value = KafkaError._QUEUE_FULL
            raise KafkaException(err)
        # third call succeeds

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer",
        lambda self: mock_producer,
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection", Mock()
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._publish", publish_side_effect
    )
    mock_retry_connection = Mock()
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._retry_connection",
        mock_retry_connection,
    )

    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer
    event = {"sender_id": "s1", "event": "user", "text": "hi"}

    broker.publish(event, retries=5)

    assert call_count[0] == 3
    mock_producer.poll.assert_called_with(1)
    mock_retry_connection.assert_not_called()


@pytest.mark.asyncio
async def test_kafka_publish_buffer_error_decrements_retries(
    monkeypatch: MonkeyPatch,
    common_kafka_config: Dict[str, Any],
) -> None:
    """publish handles BufferError: polls producer and decrements retries."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    mock_producer = MagicMock()
    mock_producer.poll.return_value = 0
    call_count = [0]

    def publish_side_effect(*args: Any, **kwargs: Any) -> None:
        call_count[0] += 1
        if call_count[0] <= 2:
            raise BufferError("Local: Queue full")
        # third call succeeds (no raise)

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer",
        lambda self: mock_producer,
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection", Mock()
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._publish", publish_side_effect
    )

    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer
    event = {"sender_id": "s1", "event": "user", "text": "hi"}

    broker.publish(event, retries=5)

    assert call_count[0] == 3
    assert mock_producer.poll.call_count >= 2
    mock_producer.poll.assert_called_with(1)


@pytest.mark.asyncio
async def test_kafka_publish_exhausts_retries_logs_error(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    common_kafka_config: Dict[str, Any],
) -> None:
    """publish logs final error when all retries are exhausted."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    mock_producer = MagicMock()
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer",
        lambda self: mock_producer,
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection", Mock()
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._publish",
        Mock(side_effect=BufferError("Queue full")),
    )

    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer
    event = {"sender_id": "s1", "event": "user", "text": "hi"}

    with caplog.at_level(logging.ERROR):
        broker.publish(event, retries=2)

    assert "Failed to publish Kafka event." in caplog.text


@pytest.mark.asyncio
async def test_kafka_publish_unexpected_exception_calls_retry_publish(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """publish calls _retry_publish and decrements retries on generic Exception."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    mock_producer = MagicMock()
    mock_retry = Mock()

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer",
        lambda self: mock_producer,
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection", Mock()
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._publish",
        Mock(side_effect=RuntimeError("unexpected")),
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._retry_connection", mock_retry
    )

    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer
    event = {"sender_id": "s1", "event": "user", "text": "hi"}

    broker.publish(event, retries=2)

    assert mock_retry.call_count == 2


@pytest.mark.asyncio
async def test_kafka_get_kafka_config_ssl(monkeypatch: MonkeyPatch) -> None:
    """_get_kafka_config returns SSL params when security_protocol is SSL."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    broker = KafkaEventBroker(
        url="localhost",
        topic="topic",
        security_protocol="SSL",
        ssl_cafile="/ca.pem",
        ssl_certfile="/cert.pem",
        ssl_keyfile="/key.pem",
    )
    config = broker._get_kafka_config()
    assert config["security.protocol"] == "SSL"
    assert config["ssl.ca.location"] == "/ca.pem"
    assert config["ssl.certificate.location"] == "/cert.pem"
    assert config["ssl.key.location"] == "/key.pem"


@pytest.mark.asyncio
async def test_kafka_get_kafka_config_sasl_ssl_without_iam(
    monkeypatch: MonkeyPatch,
) -> None:
    """_get_kafka_config with SASL_SSL and no IAM uses sasl username/password and ssl certs."""  # noqa: E501
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker.iam_credentials_provider",
        None,
        raising=False,
    )
    broker = KafkaEventBroker(
        url="localhost",
        topic="topic",
        sasl_username="u",
        sasl_password="p",
        sasl_mechanism="PLAIN",
        security_protocol="SASL_SSL",
        ssl_cafile="/ca.pem",
        ssl_certfile="/cert.pem",
        ssl_keyfile="/key.pem",
    )
    # Replace the provider that was set in __init__
    broker.iam_credentials_provider = None
    config = broker._get_kafka_config()
    assert config["security.protocol"] == "SASL_SSL"
    assert config["sasl.mechanism"] == "PLAIN"
    assert config["sasl.username"] == "u"
    assert config["sasl.password"] == "p"
    assert "oauth_cb" not in config


@pytest.mark.asyncio
async def test_kafka_get_kafka_config_invalid_protocol_raises(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """_get_kafka_config raises ValueError for invalid security_protocol."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.security_protocol = "INVALID"
    with pytest.raises(ValueError, match="Invalid.*security_protocol"):
        broker._get_kafka_config()


@pytest.mark.asyncio
async def test_kafka_get_kafka_config_queue_size(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """_get_kafka_config includes queue.buffering.max.messages when queue_size set."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
        queue_size=100,
    )
    config = broker._get_kafka_config()
    assert config["queue.buffering.max.messages"] == 100


@pytest.mark.asyncio
async def test_kafka_publish_partition_key_none_when_not_by_sender(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """_publish uses partition_key None when partition_by_sender is False."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.join", lambda self, timeout: None
    )
    mock_producer = MagicMock()
    broker = KafkaEventBroker(
        **common_kafka_config,
        partition_by_sender=False,
    )
    broker.producer = mock_producer
    event = {"sender_id": "user123", "event": "user", "text": "hello"}

    try:
        broker._publish(event)

        call_kw = mock_producer.produce.call_args[1]
        assert call_kw["key"] is None
    finally:
        await broker.close()


@pytest.mark.asyncio
async def test_kafka_get_aws_iam_token_returns_none_when_no_provider(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """get_aws_iam_token returns (None, None) when iam_credentials_provider is None."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.iam_credentials_provider = None
    assert broker.get_aws_iam_token(None) == (None, None)


@pytest.mark.asyncio
async def test_kafka_get_aws_iam_token_returns_credentials_from_provider(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """get_aws_iam_token returns (token, expiration) from credentials provider."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    creds = TemporaryCredentials(auth_token="tok", expiration=123.0)
    mock_provider = MagicMock()
    mock_provider.get_temporary_credentials.return_value = creds
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.iam_credentials_provider = mock_provider

    token, exp = broker.get_aws_iam_token("oauth_config")

    assert token == "tok"
    assert exp == 123.0
    mock_provider.get_temporary_credentials.assert_called_once()


@pytest.mark.asyncio
async def test_kafka_rasa_environment_from_env(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """rasa_environment cached property returns RASA_ENVIRONMENT env var."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    monkeypatch.setenv("RASA_ENVIRONMENT", "my-env")
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    assert broker.rasa_environment == "my-env"


@pytest.mark.asyncio
async def test_kafka_close_flushes_producer(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """close sets _cancelled, joins poll thread, and flushes producer."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.join", lambda self, timeout: None
    )
    mock_producer = MagicMock()
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer

    await broker.close()

    assert broker._cancelled is True
    mock_producer.flush.assert_called_once()


@pytest.mark.asyncio
async def test_kafka_error_callback_raises_for_brokers_down(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """error_cb from broker config raises KafkaException for _ALL_BROKERS_DOWN."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.join", lambda self, timeout: None
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    config = broker._get_kafka_config()
    error_cb = config["error_cb"]

    err = MagicMock()
    err.code.return_value = KafkaError._ALL_BROKERS_DOWN
    with pytest.raises(KafkaException):
        error_cb(err)


@pytest.mark.asyncio
async def test_kafka_error_callback_logs_for_other_errors(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    common_kafka_config: Dict[str, Any],
) -> None:
    """error_cb from broker config logs warning for non-fatal error codes."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    config = broker._get_kafka_config()
    error_cb = config["error_cb"]

    err = MagicMock()
    err.code.return_value = KafkaError.MSG_SIZE_TOO_LARGE  # not in raise list
    with caplog.at_level(logging.WARNING):
        error_cb(err)
    assert "KafkaError" in caplog.text


@pytest.mark.asyncio
async def test_kafka_error_callback_invoked_during_close_when_flush_reports_error(
    monkeypatch: MonkeyPatch,
    common_kafka_config: Dict[str, Any],
) -> None:
    """error_cb is invoked during close() when flush() triggers an error (normal flow)."""  # noqa: E501
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.join", lambda self, timeout: None
    )
    err_all_brokers_down = MagicMock()
    err_all_brokers_down.code.return_value = KafkaError._ALL_BROKERS_DOWN

    class ProducerThatCallsErrorCbOnFlush:
        def __init__(self, config: dict) -> None:
            self._config = config

        def poll(self, timeout: float = 0) -> int:
            return 0

        def produce(self, *args: Any, **kwargs: Any) -> None:
            pass

        def list_topics(self, topic: Optional[str] = None, timeout: int = 5) -> Any:
            pass

        def flush(self) -> None:
            self._config["error_cb"](err_all_brokers_down)

    monkeypatch.setattr(
        "confluent_kafka.Producer",
        ProducerThatCallsErrorCbOnFlush,
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = ProducerThatCallsErrorCbOnFlush(broker._get_kafka_config())

    with pytest.raises(KafkaException):
        await broker.close()


@pytest.mark.asyncio
async def test_kafka_delivery_report_invoked_during_publish_when_poll_simulates_delivery(  # noqa: E501
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    common_kafka_config: Dict[str, Any],
) -> None:
    """on_delivery is invoked during publish when poll() runs (simulates normal delivery)."""  # noqa: E501
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    pending_deliveries: list = []

    class ProducerThatCallsOnDeliveryOnPoll:
        def __init__(self, config: dict) -> None:
            self._config = config

        def produce(self, *args: Any, **kwargs: Any) -> None:
            on_delivery = kwargs.get("on_delivery")
            if on_delivery is not None:
                pending_deliveries.append(on_delivery)

        def poll(self, timeout: float = 0) -> int:
            if pending_deliveries:
                cb = pending_deliveries.pop(0)
                msg = MagicMock()
                msg.key.return_value = b"key"
                msg.topic.return_value = "topic"
                msg.partition.return_value = 0
                msg.offset.return_value = 42
                cb(None, msg)
            return 0

        def list_topics(self, topic: Optional[str] = None, timeout: int = 5) -> Any:
            pass

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection", Mock()
    )
    monkeypatch.setattr(
        "confluent_kafka.Producer",
        ProducerThatCallsOnDeliveryOnPoll,
    )
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    event = {"sender_id": "s1", "event": "user", "text": "hi"}

    with caplog.at_level(logging.INFO):
        broker.publish(event)
        assert broker.producer is not None
        broker.producer.poll(0)

    assert "successfully produced" in caplog.text


@pytest.mark.asyncio
async def test_kafka_delivery_report_logs_error_when_err_set(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    common_kafka_config: Dict[str, Any],
) -> None:
    """on_delivery callback passed to produce() logs error when err is not None."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    produce_kwargs: dict = {}

    def capture_produce(*args: Any, **kwargs: Any) -> None:
        produce_kwargs.clear()
        produce_kwargs.update(kwargs)

    mock_producer = MagicMock()
    mock_producer.produce.side_effect = capture_produce
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer

    broker._publish({"sender_id": "s1", "event": "user", "text": "hi"})

    on_delivery = produce_kwargs.get("on_delivery")
    assert on_delivery is not None
    msg = MagicMock()
    msg.key.return_value = b"key"
    with caplog.at_level(logging.ERROR):
        on_delivery(Exception("delivery failed"), msg)
    assert "Delivery failed" in caplog.text


@pytest.mark.asyncio
async def test_kafka_delivery_report_logs_success_when_err_none(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    common_kafka_config: Dict[str, Any],
) -> None:
    """on_delivery callback passed to produce() logs info when err is None (success)."""
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    produce_kwargs: dict = {}

    def capture_produce(*args: Any, **kwargs: Any) -> None:
        produce_kwargs.clear()
        produce_kwargs.update(kwargs)

    mock_producer = MagicMock()
    mock_producer.produce.side_effect = capture_produce
    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = mock_producer

    broker._publish({"sender_id": "s1", "event": "user", "text": "hi"})

    on_delivery = produce_kwargs.get("on_delivery")
    assert on_delivery is not None
    msg = MagicMock()
    msg.key.return_value = b"key"
    msg.topic.return_value = "t"
    msg.partition.return_value = 0
    msg.offset.return_value = 42
    with caplog.at_level(logging.INFO):
        on_delivery(None, msg)
    assert "successfully produced" in caplog.text


@pytest.mark.asyncio
async def test_kafka_retry_publish_reconnects_on_connection_failure(
    monkeypatch: MonkeyPatch, common_kafka_config: Dict[str, Any]
) -> None:
    """_retry_publish recreates producer and retries when _check_kafka_connection fails."""  # noqa: E501
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.threading.Thread.start", lambda self: None
    )
    mock_producer = MagicMock()
    mock_create = Mock(return_value=mock_producer)
    check_calls = [0]

    def check_side_effect() -> None:
        check_calls[0] += 1
        if check_calls[0] == 1:
            raise KafkaException(KafkaError(1))
        # second call (after reconnect) succeeds

    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._create_producer", mock_create
    )
    monkeypatch.setattr(
        "rasa.core.brokers.kafka.KafkaEventBroker._check_kafka_connection",
        Mock(side_effect=check_side_effect),
    )
    monkeypatch.setattr("rasa.core.brokers.kafka.time.sleep", Mock())

    broker = KafkaEventBroker(
        **common_kafka_config,
    )
    broker.producer = MagicMock()

    broker._retry_connection(retry_delay_in_seconds=1)

    assert check_calls[0] == 2
    mock_create.assert_called_once()
    assert broker.producer == mock_producer
