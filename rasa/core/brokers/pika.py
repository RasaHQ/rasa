import json
import logging
import typing
from typing import Dict, Optional, Text, Union

import time

from rasa.core.brokers.event_channel import EventChannel
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection

logger = logging.getLogger(__name__)


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
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        Pika `BlockingConnection` with provided parameters
    """

    import pika

    if host.startswith("amqp"):
        # user supplied a amqp url containing all the info
        parameters = pika.URLParameters(host)
        parameters.connection_attempts = connection_attempts
        parameters.retry_delay = retry_delay_in_seconds
        if username:
            parameters.credentials = pika.PlainCredentials(username, password)
    else:
        # host seems to be just the host, so we use our parameters
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


def initialise_pika_channel(
    host: Text,
    queue: Text,
    username: Text,
    password: Text,
    connection_attempts: int = 20,
    retry_delay_in_seconds: Union[int, float] = 5,
) -> "BlockingChannel":
    """Initialise a Pika channel with a durable queue.

    Args:
        host: Pika host
        queue: Pika queue to declare
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        Pika `BlockingChannel` with declared queue
    """

    connection = initialise_pika_connection(
        host, username, password, connection_attempts, retry_delay_in_seconds
    )

    return _declare_pika_channel_with_queue(connection, queue)


def _declare_pika_channel_with_queue(
    connection: "BlockingConnection", queue: Text
) -> "BlockingChannel":
    """Declare a durable queue on Pika channel."""

    channel = connection.channel()
    channel.queue_declare(queue, durable=True)

    return channel


def close_pika_channel(channel: "BlockingChannel") -> None:
    """Attempt to close Pika channel."""

    from pika.exceptions import AMQPError

    try:
        channel.close()
        logger.debug("Successfully closed Pika channel.")
    except AMQPError:
        logger.exception("Failed to close Pika channel.")


def close_pika_connection(connection: "BlockingConnection") -> None:
    """Attempt to close Pika connection."""

    from pika.exceptions import AMQPError

    try:
        connection.close()
        logger.debug("Successfully closed Pika connection with host.")
    except AMQPError:
        logger.exception("Failed to close Pika connection with host.")


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
        self.channel = None  # delay opening channel until first event

    def __del__(self) -> None:
        if self.channel:
            close_pika_channel(self.channel)
            close_pika_connection(self.channel.connection)

    def _open_channel(
        self,
        connection_attempts: int = 20,
        retry_delay_in_seconds: Union[int, float] = 5,
    ) -> "BlockingChannel":
        return initialise_pika_channel(
            self.host,
            self.queue,
            self.username,
            self.password,
            connection_attempts,
            retry_delay_in_seconds,
        )

    @classmethod
    def from_endpoint_config(
        cls, broker_config: Optional["EndpointConfig"]
    ) -> Optional["PikaProducer"]:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(self, event: Dict, retries=60, retry_delay_in_seconds: int = 5) -> None:
        """Publish `event` into Pika queue.

        Perform `retries` publish attempts with `retry_delay_in_seconds` between them.
        """

        body = json.dumps(event)

        while retries:
            # noinspection PyBroadException
            try:
                self._publish(body)
                return
            except Exception as e:
                logger.error(
                    "Could not open Pika channel at host '{}'. Failed with error: "
                    "{}".format(self.host, e)
                )
                self.channel = None

            retries -= 1
            time.sleep(retry_delay_in_seconds)

        logger.error(
            "Failed to publish Pika event to queue '{}' on host "
            "'{}':\n{}".format(self.queue, self.host, body)
        )

    def _publish(self, body: Text) -> None:
        if not self.channel:
            self.channel = self._open_channel(connection_attempts=1)

        self.channel.basic_publish("", self.queue, body)

        logger.debug(
            "Published Pika events to queue '{}' on host "
            "'{}':\n{}".format(self.queue, self.host, body)
        )
