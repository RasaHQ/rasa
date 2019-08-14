import json
import logging
import typing
from typing import Any, Dict, Optional, Text, Union

import rasa.utils.common as rasa_utils
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection

logger = logging.getLogger(__name__)


def from_endpoint_config(
    broker_config: Optional[EndpointConfig]
) -> Optional["EventChannel"]:
    """Instantiate an event channel based on its configuration."""

    if broker_config is None:
        return None
    elif broker_config.type == "pika" or broker_config.type is None:
        return PikaProducer.from_endpoint_config(broker_config)
    elif broker_config.type.lower() == "sql":
        return SQLProducer.from_endpoint_config(broker_config)
    elif broker_config.type == "file":
        return FileProducer.from_endpoint_config(broker_config)
    elif broker_config.type == "kafka":
        return KafkaProducer.from_endpoint_config(broker_config)
    else:
        return load_event_channel_from_module_string(broker_config)


def load_event_channel_from_module_string(
    broker_config: EndpointConfig
) -> Optional["EventChannel"]:
    """Instantiate an event channel based on its class name."""

    try:
        event_channel = rasa_utils.class_from_module_path(broker_config.type)
        return event_channel.from_endpoint_config(broker_config)
    except (AttributeError, ImportError) as e:
        logger.warning(
            "EventChannel type '{}' not found. "
            "Not using any event channel. Error: {}".format(broker_config.type, e)
        )
        return None


class EventChannel(object):
    @classmethod
    def from_endpoint_config(cls, broker_config: EndpointConfig) -> "EventChannel":
        raise NotImplementedError(
            "Event broker must implement the `from_endpoint_config` method."
        )

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""

        raise NotImplementedError("Event broker must implement the `publish` method.")


# noinspection PyUnresolvedReferences
def initialise_pika_connection(
    host: Text,
    username: Text,
    password: Text,
    connection_attempts: int = 20,
    retry_delay_in_seconds: Union[int, float] = 5,
) -> "BlockingConnection":
    """Create a Pika `BlockingConnection`.

    Args:
        host: Pika host
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        connection_attempts: number of connection attempts before giving up
        retry_delay_in_seconds: delay in seconds between connection attempts

    Returns:
        Pika `BlockingConnection` with provided parameters
    """

    import pika

    parameters = pika.ConnectionParameters(
        host,
        credentials=pika.PlainCredentials(username, password),
        connection_attempts=connection_attempts,
        # Wait between retries since
        # it can take some time until
        # RabbitMQ comes up.
        retry_delay=retry_delay_in_seconds,
    )
    return pika.BlockingConnection(parameters)


# noinspection PyUnresolvedReferences
def initialise_pika_channel(
    host: Text, queue: Text, username: Text, password: Text
) -> "BlockingChannel":
    """Initialise a Pika channel with a durable queue.

    Args:
        host: Pika host
        queue: Pika queue to declare
        username: username for authentication with Pika host
        password: password for authentication with Pika host

    Returns:
        Pika `BlockingChannel` with declared queue
    """

    connection = initialise_pika_connection(host, username, password)

    return _declare_pika_channel_with_queue(connection, queue)


# noinspection PyUnresolvedReferences
def _declare_pika_channel_with_queue(
    connection: "BlockingConnection", queue: Text
) -> "BlockingChannel":
    """Declare a durable queue on Pika connection."""

    channel = connection.channel()
    channel.queue_declare(queue, durable=True)

    return channel


# noinspection PyUnresolvedReferences
def close_pika_connection(connection: "BlockingConnection") -> None:
    """Attempt to close Pika connection."""

    from pika.exceptions import AMQPError

    host = connection.parameters.host
    try:
        connection.close()
        logger.debug("Successfully closed Pika connection with host '{}'.".format(host))
    except AMQPError:
        logger.exception("Failed to close Pika connection with host '{}'.".format(host))


class PikaProducer(EventChannel):
    def __init__(
        self,
        host: Text,
        username: Text,
        password: Text,
        queue: Text = "rasa_core_events",
        loglevel: Union[Text, int] = logging.WARNING,
    ):
        logging.getLogger("pika").setLevel(loglevel)

        self.queue = queue
        self.host = host
        self.username = username
        self.password = password

    @rasa_utils.lazyproperty
    def channel(self):
        return initialise_pika_channel(
            self.host, self.queue, self.username, self.password
        )

    @classmethod
    def from_endpoint_config(
        cls, broker_config: Optional["EndpointConfig"]
    ) -> Optional["PikaProducer"]:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(self, event: Dict) -> None:
        body = json.dumps(event)
        self.channel.basic_publish("", self.queue, body)
        logger.debug(
            "Published Pika events to queue '{}' on host "
            "'{}':\n{}".format(self.queue, self.host, body)
        )


class FileProducer(EventChannel):
    """Log events to a file in json format.

    There will be one event per line and each event is stored as json."""

    DEFAULT_LOG_FILE_NAME = "rasa_event.log"

    def __init__(self, path: Optional[Text] = None) -> None:
        self.path = path or self.DEFAULT_LOG_FILE_NAME
        self.event_logger = self._event_logger()

    @classmethod
    def from_endpoint_config(
        cls, broker_config: Optional["EndpointConfig"]
    ) -> Optional["FileProducer"]:
        if broker_config is None:
            return None

        # noinspection PyArgumentList
        return cls(**broker_config.kwargs)

    def _event_logger(self):
        """Instantiate the file logger."""

        logger_file = self.path
        # noinspection PyTypeChecker
        query_logger = logging.getLogger("event-logger")
        query_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logger_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        query_logger.propagate = False
        query_logger.addHandler(handler)

        logger.info("Logging events to '{}'.".format(logger_file))

        return query_logger

    def publish(self, event: Dict) -> None:
        """Write event to file."""

        self.event_logger.info(json.dumps(event))
        self.event_logger.handlers[0].flush()


class KafkaProducer(EventChannel):
    def __init__(
        self,
        host,
        sasl_username=None,
        sasl_password=None,
        ssl_cafile=None,
        ssl_certfile=None,
        ssl_keyfile=None,
        ssl_check_hostname=False,
        topic="rasa_core_events",
        security_protocol="SASL_PLAINTEXT",
        loglevel=logging.ERROR,
    ):

        self.producer = None
        self.host = host
        self.topic = topic
        self.security_protocol = security_protocol
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_check_hostname = ssl_check_hostname

        logging.getLogger("kafka").setLevel(loglevel)

    @classmethod
    def from_endpoint_config(cls, broker_config) -> Optional["KafkaProducer"]:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(self, event):
        self._create_producer()
        self._publish(event)
        self._close()

    def _create_producer(self):
        import kafka

        if self.security_protocol == "SASL_PLAINTEXT":
            self.producer = kafka.KafkaProducer(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                sasl_plain_username=self.sasl_username,
                sasl_plain_password=self.sasl_password,
                sasl_mechanism="PLAIN",
                security_protocol=self.security_protocol,
            )
        elif self.security_protocol == "SSL":
            self.producer = kafka.KafkaProducer(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                ssl_cafile=self.ssl_cafile,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
                ssl_check_hostname=False,
                security_protocol=self.security_protocol,
            )

    def _publish(self, event):
        self.producer.send(self.topic, event)

    def _close(self):
        self.producer.close()


class SQLProducer(EventChannel):
    """Save events into an SQL database.

    All events will be stored in a table called `events`.

    """

    from sqlalchemy.ext.declarative import declarative_base

    Base = declarative_base()

    class SQLBrokerEvent(Base):
        from sqlalchemy import Column, Integer, String, Text

        __tablename__ = "events"
        id = Column(Integer, primary_key=True)
        sender_id = Column(String(255))
        data = Column(Text)

    def __init__(
        self,
        dialect: Text = "sqlite",
        host: Optional[Text] = None,
        port: Optional[int] = None,
        db: Text = "events.db",
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ):
        from rasa.core.tracker_store import SQLTrackerStore
        import sqlalchemy.orm

        engine_url = SQLTrackerStore.get_db_url(
            dialect, host, port, db, username, password
        )

        logger.debug("SQLProducer: Connecting to database: '{}'.".format(engine_url))

        self.engine = sqlalchemy.create_engine(engine_url)
        self.Base.metadata.create_all(self.engine)
        self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)()

    @classmethod
    def from_endpoint_config(cls, broker_config: EndpointConfig) -> "EventChannel":
        return cls(host=broker_config.url, **broker_config.kwargs)

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""
        self.session.add(
            self.SQLBrokerEvent(
                sender_id=event.get("sender_id"), data=json.dumps(event)
            )
        )
        self.session.commit()
