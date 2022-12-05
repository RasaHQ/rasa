import asyncio
import json
import logging
import os
import ssl
from asyncio import AbstractEventLoop
from collections import deque
from typing import Deque, Dict, Optional, Text, Union, Any, List, Tuple, cast
from urllib.parse import urlparse

import aio_pika

from rasa.shared.exceptions import RasaException
from rasa.shared.constants import DOCS_URL_PIKA_EVENT_BROKER
from rasa.core.brokers.broker import EventBroker
import rasa.shared.utils.io
from rasa.utils.endpoints import EndpointConfig
from rasa.shared.utils.io import DEFAULT_ENCODING
import rasa.shared.utils.common

logger = logging.getLogger(__name__)

RABBITMQ_EXCHANGE = "rasa-exchange"
DEFAULT_QUEUE_NAME = "rasa_core_events"


class PikaEventBroker(EventBroker):
    """Pika-based event broker for publishing messages to RabbitMQ."""

    def __init__(
        self,
        host: Text,
        username: Text,
        password: Text,
        port: Union[int, Text] = 5672,
        queues: Union[List[Text], Tuple[Text, ...], Text, None] = None,
        should_keep_unpublished_messages: bool = True,
        raise_on_failure: bool = False,
        event_loop: Optional[AbstractEventLoop] = None,
        connection_attempts: int = 20,
        retry_delay_in_seconds: float = 5,
        exchange_name: Text = RABBITMQ_EXCHANGE,
        **kwargs: Any,
    ):
        """Initialise RabbitMQ event broker.

        Args:
            host: Pika host.
            username: Username for authentication with Pika host.
            password: Password for authentication with Pika host.
            port: port of the Pika host.
            queues: Pika queues to declare and publish to.
            should_keep_unpublished_messages: Whether or not the event broker should
                maintain a queue of unpublished messages to be published later in
                case of errors.
            raise_on_failure: Whether to raise an exception if publishing fails. If
                `False`, keep retrying.
            event_loop: The event loop which will be used to run `async` functions. If
                `None` `asyncio.get_event_loop()` is used to get a loop.
            connection_attempts: Number of attempts for connecting to RabbitMQ before
                an exception is thrown.
            retry_delay_in_seconds: Time in seconds between connection attempts.
            exchange_name: Exchange name to which the queues binds to.
                If nothing is mentioned then the default exchange name would be used.
        """
        super().__init__()

        self.host = host
        self.username = username
        self.password = password

        try:
            self.port = int(port)
        except ValueError as e:
            raise RasaException("Port could not be converted to integer.") from e

        self.queues = self._get_queues_from_args(queues)
        self.raise_on_failure = raise_on_failure
        self._connection_attempts = connection_attempts
        self._retry_delay_in_seconds = retry_delay_in_seconds
        self.exchange_name = exchange_name

        # Unpublished messages which hopefully will be published later 🤞
        self._unpublished_events: Deque[Dict[Text, Any]] = deque()
        self.should_keep_unpublished_messages = should_keep_unpublished_messages

        self._loop = event_loop or asyncio.get_event_loop()

        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self._exchange: Optional[aio_pika.RobustExchange] = None

    @staticmethod
    def _get_queues_from_args(
        queues_arg: Union[List[Text], Tuple[Text, ...], Text, None]
    ) -> Union[List[Text], Tuple[Text, ...]]:
        """Get queues for this event broker.

        The preferred argument defining the RabbitMQ queues the `PikaEventBroker` should
        publish to is `queues` (as of Rasa Open Source version 1.8.2). This method
        can be removed in the future, and `self.queues` should just receive the value of
        the `queues` kwarg in the constructor.

        Args:
            queues_arg: Value of the supplied `queues` argument.

        Returns:
            Queues this event broker publishes to.

        Raises:
            `ValueError` if no valid `queues` argument was found.
        """
        if queues_arg and isinstance(queues_arg, (list, tuple)):
            return queues_arg

        if queues_arg and isinstance(queues_arg, str):
            logger.debug(
                f"Found a string value under the `queues` key of the Pika event broker "
                f"config. Please supply a list of queues under this key, even if it is "
                f"just a single one. See {DOCS_URL_PIKA_EVENT_BROKER}"
            )
            return [queues_arg]

        rasa.shared.utils.io.raise_warning(
            f"No `queues` argument provided. It is suggested to "
            f"explicitly specify a queue as described in "
            f"{DOCS_URL_PIKA_EVENT_BROKER}. "
            f"Using the default queue '{DEFAULT_QUEUE_NAME}' for now."
        )

        return [DEFAULT_QUEUE_NAME]

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: Optional["EndpointConfig"],
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> Optional["PikaEventBroker"]:
        """Creates broker. See the parent class for more information."""
        if broker_config is None:
            return None

        broker = cls(broker_config.url, **broker_config.kwargs, event_loop=event_loop)
        await broker.connect()

        return broker

    async def connect(self) -> None:
        """Connects to RabbitMQ."""
        self._connection = await self._connect()
        self._connection.reconnect_callbacks.add(self._publish_unpublished_messages)
        logger.info(f"RabbitMQ connection to '{self.host}' was established.")

        channel = await self._connection.channel()
        logger.debug(
            f"RabbitMQ channel was opened. "
            f"Declaring fanout exchange '{self.exchange_name}'."
        )

        self._exchange = await self._set_up_exchange(channel)

    async def _connect(self) -> aio_pika.abc.AbstractRobustConnection:
        url = None
        # The `url` parameter will take precedence over parameters like `login` or
        # `password`.
        if self.host.startswith("amqp"):

            parsed_host = urlparse(self.host)
            amqp_user = f"{self.username}:{self.password}"
            url = f"{parsed_host.scheme}://{amqp_user}@{parsed_host.netloc}:{self.port}"

        ssl_options = _create_rabbitmq_ssl_options(self.host)
        logger.info("Connecting to RabbitMQ ...")

        last_exception: Optional[Exception] = None
        for _ in range(self._connection_attempts):
            try:
                return await aio_pika.connect_robust(
                    url=url,
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    login=self.username,
                    loop=self._loop,
                    ssl=ssl_options is not None,
                    ssl_options=ssl_options,
                )
            # All sorts of exception can happen until RabbitMQ is in a stable state
            except Exception as e:
                last_exception = e
                logger.debug(
                    f"Connecting to '{self.host}' failed with error '{e}'. "
                    f"Trying again."
                )
                await asyncio.sleep(self._retry_delay_in_seconds)

        last_exception = cast(Exception, last_exception)
        logger.error(
            f"Connecting to '{self.host}' failed with error '{last_exception}'."
        )
        raise last_exception

    def _publish_unpublished_messages(self, *_: Any, **__: Any) -> None:
        while self._unpublished_events:
            # Send unpublished messages
            message = self._unpublished_events.popleft()
            self.publish(message)
            logger.debug(
                f"Published message from queue of unpublished messages. "
                f"Remaining unpublished messages: {len(self._unpublished_events)}."
            )

    async def _set_up_exchange(
        self, channel: aio_pika.RobustChannel
    ) -> aio_pika.RobustExchange:
        exchange = await channel.declare_exchange(
            self.exchange_name, type=aio_pika.ExchangeType.FANOUT
        )

        await asyncio.gather(
            *[
                self._bind_queue(queue_name, channel, exchange)
                for queue_name in self.queues
            ]
        )

        return exchange

    @staticmethod
    async def _bind_queue(
        queue_name: Text, channel: aio_pika.RobustChannel, exchange: aio_pika.Exchange
    ) -> None:
        queue = await channel.declare_queue(queue_name, durable=True)

        await queue.bind(exchange, "")

    async def close(self) -> None:
        """Closes connection to RabbitMQ."""
        if not self._connection:
            return

        # Entering the context manager does nothing. Exiting closes the channels and
        # the connection.
        async with self._connection:
            logger.debug("Closing RabbitMQ connection.")

    def is_ready(self) -> bool:
        """Return `True` if a connection was established."""
        if self._connection is None:
            return False

        return not self._connection.is_closed

    def publish(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]] = None
    ) -> None:
        """Publishes `event` to Pika queues.

        Args:
            event: Serialised event to be published.
            headers: Message headers to append to the published message. The headers
                can be retrieved in the consumer from the `headers` attribute of the
                message's `BasicProperties`.
        """
        self._loop.create_task(self._publish(event, headers))

    async def _publish(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]] = None
    ) -> None:
        if self._exchange is None:
            return

        try:
            await self._exchange.publish(self._message(event, headers), "")

            logger.debug(
                f"Published Pika events to exchange '{RABBITMQ_EXCHANGE}' on host "
                f"'{self.host}':\n{event}"
            )
        except Exception as e:
            logger.error(
                f"Failed to publish Pika event on host '{self.host}' due to "
                f"error '{e}'. The message was: \n{event}"
            )
            if self.should_keep_unpublished_messages:
                self._unpublished_events.append(event)

            if self.raise_on_failure:
                self.close()  # type: ignore[unused-coroutine]
                raise e

    def _message(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]]
    ) -> aio_pika.Message:
        body = json.dumps(event)
        return aio_pika.Message(
            bytes(body, DEFAULT_ENCODING),
            headers=headers,
            app_id=self.rasa_environment,
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

    @rasa.shared.utils.common.lazy_property
    def rasa_environment(self) -> Optional[Text]:
        """Get value of the `RASA_ENVIRONMENT` environment variable."""
        return os.environ.get("RASA_ENVIRONMENT")


def _create_rabbitmq_ssl_options(
    rabbitmq_host: Optional[Text] = None,
) -> Optional[Dict]:
    """Create RabbitMQ SSL options.

    Requires the following environment variables to be set:

        RABBITMQ_SSL_CLIENT_CERTIFICATE - path to the SSL client certificate (required)
        RABBITMQ_SSL_CLIENT_KEY - path to the SSL client key (required)

    Details on how to enable RabbitMQ TLS support can be found here:
    https://www.rabbitmq.com/ssl.html#enabling-tls

    Args:
        rabbitmq_host: RabbitMQ hostname.

    Returns:
        Optional SSL arguments for the RabbitMQ connection.
    """
    client_certificate_path = os.environ.get("RABBITMQ_SSL_CLIENT_CERTIFICATE")
    client_key_path = os.environ.get("RABBITMQ_SSL_CLIENT_KEY")

    if os.environ.get("RABBITMQ_SSL_CA_FILE"):
        rasa.shared.utils.io.raise_warning(
            "Specifying 'RABBITMQ_SSL_CA_FILE' via "
            "environment variables is no longer supported. Please specify this "
            "through the RabbitMQ URL parameter 'cacertfile' as described here: "
            "https://www.rabbitmq.com/uri-query-parameters.html "
        )

    if os.environ.get("RABBITMQ_SSL_KEY_PASSWORD"):
        rasa.shared.utils.io.raise_warning(
            "Specifying 'RABBITMQ_SSL_KEY_PASSWORD' via environment variables is no "
            "longer supported. Please use an unencrypted key file."
        )

    if client_certificate_path and client_key_path:
        logger.debug(f"Configuring SSL context for RabbitMQ host '{rabbitmq_host}'.")
        return {
            "certfile": client_certificate_path,
            "client_key_path": client_key_path,
            "cert_reqs": ssl.CERT_REQUIRED,
        }

    return None
