import json
import logging
import os
import sys
import time
import typing
import threading
import multiprocessing
from contextlib import contextmanager
from typing import (
    Callable,
    Dict,
    Optional,
    Text,
    Union,
    Any,
    List,
    Tuple,
    Generator,
)

from rasa.constants import (
    DEFAULT_LOG_LEVEL_LIBRARIES,
    ENV_LOG_LEVEL_LIBRARIES,
    DOCS_URL_PIKA_EVENT_BROKER,
)
from rasa.core.brokers.broker import EventBroker
from rasa.utils.common import raise_warning
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.io import DEFAULT_ENCODING

if typing.TYPE_CHECKING:
    from pika.adapters.blocking_connection import BlockingChannel
    from pika import SelectConnection, BlockingConnection, BasicProperties
    from pika.channel import Channel
    import pika
    from pika.connection import Parameters, Connection

logger = logging.getLogger(__name__)

RABBITMQ_EXCHANGE = "rasa-exchange"
DEFAULT_QUEUE_NAME = "rasa_core_events"


def initialise_pika_connection(
    host: Text,
    username: Text,
    password: Text,
    port: Union[Text, int] = 5672,
    connection_attempts: int = 20,
    retry_delay_in_seconds: float = 5,
) -> "BlockingConnection":
    """Create a Pika `BlockingConnection`.

    Args:
        host: Pika host
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        port: port of the Pika host
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        `pika.BlockingConnection` with provided parameters
    """
    import pika

    with _pika_log_level(logging.CRITICAL):
        parameters = _get_pika_parameters(
            host, username, password, port, connection_attempts, retry_delay_in_seconds
        )
        return pika.BlockingConnection(parameters)


@contextmanager
def _pika_log_level(temporary_log_level: int) -> Generator[None, None, None]:
    """Change the log level of the `pika` library.

    The log level will remain unchanged if the current log level is 10 (`DEBUG`) or
    lower.

    Args:
        temporary_log_level: Temporary log level for pika. Will be reverted to
        previous log level when context manager exits.
    """
    pika_logger = logging.getLogger("pika")
    old_log_level = pika_logger.level
    is_debug_mode = logging.root.level <= logging.DEBUG

    if not is_debug_mode:
        pika_logger.setLevel(temporary_log_level)

    yield

    pika_logger.setLevel(old_log_level)


def create_rabbitmq_ssl_options(
    rabbitmq_host: Optional[Text] = None,
) -> Optional["pika.SSLOptions"]:
    """Create RabbitMQ SSL options.

    Requires the following environment variables to be set:

        RABBITMQ_SSL_CLIENT_CERTIFICATE - path to the SSL client certificate (required)
        RABBITMQ_SSL_CLIENT_KEY - path to the SSL client key (required)
        RABBITMQ_SSL_CA_FILE - path to the SSL CA file for verification (optional)
        RABBITMQ_SSL_KEY_PASSWORD - SSL private key password (optional)

    Details on how to enable RabbitMQ TLS support can be found here:
    https://www.rabbitmq.com/ssl.html#enabling-tls

    Args:
        rabbitmq_host: RabbitMQ hostname

    Returns:
        Pika SSL context of type `pika.SSLOptions` if
        the RABBITMQ_SSL_CLIENT_CERTIFICATE and RABBITMQ_SSL_CLIENT_KEY
        environment variables are valid paths, else `None`.
    """
    client_certificate_path = os.environ.get("RABBITMQ_SSL_CLIENT_CERTIFICATE")
    client_key_path = os.environ.get("RABBITMQ_SSL_CLIENT_KEY")

    if client_certificate_path and client_key_path:
        import pika
        import rasa.server

        logger.debug(f"Configuring SSL context for RabbitMQ host '{rabbitmq_host}'.")

        ca_file_path = os.environ.get("RABBITMQ_SSL_CA_FILE")
        key_password = os.environ.get("RABBITMQ_SSL_KEY_PASSWORD")

        ssl_context = rasa.server.create_ssl_context(
            client_certificate_path, client_key_path, ca_file_path, key_password
        )
        return pika.SSLOptions(ssl_context, rabbitmq_host)
    else:
        return None


def _get_pika_parameters(
    host: Text,
    username: Text,
    password: Text,
    port: Union[Text, int] = 5672,
    connection_attempts: int = 20,
    retry_delay_in_seconds: float = 5,
) -> "Parameters":
    """Create Pika `Parameters`.

    Args:
        host: Pika host
        username: username for authentication with Pika host
        password: password for authentication with Pika host
        port: port of the Pika host
        connection_attempts: number of channel attempts before giving up
        retry_delay_in_seconds: delay in seconds between channel attempts

    Returns:
        `pika.ConnectionParameters` which can be used to create a new connection to a
        broker.
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
            port=port,
            credentials=pika.PlainCredentials(username, password),
            connection_attempts=connection_attempts,
            # Wait between retries since
            # it can take some time until
            # RabbitMQ comes up.
            retry_delay=retry_delay_in_seconds,
            ssl_options=create_rabbitmq_ssl_options(host),
        )

    return parameters


def initialise_pika_select_connection(
    parameters: "Parameters",
    on_open_callback: Callable[["SelectConnection"], None],
    on_open_error_callback: Callable[["SelectConnection", Text], None],
) -> "SelectConnection":
    """Create a non-blocking Pika `SelectConnection`.

    Args:
        parameters: Parameters which should be used to connect.
        on_open_callback: Callback which is called when the connection was established.
        on_open_error_callback: Callback which is called when connecting to the broker
            failed.

    Returns:
        A callback-based connection to the RabbitMQ event broker.
    """
    import pika

    return pika.SelectConnection(
        parameters,
        on_open_callback=on_open_callback,
        on_open_error_callback=on_open_error_callback,
    )


def initialise_pika_channel(
    host: Text,
    queue: Text,
    username: Text,
    password: Text,
    port: Union[Text, int] = 5672,
    connection_attempts: int = 20,
    retry_delay_in_seconds: float = 5,
) -> "BlockingChannel":
    """Initialise a Pika channel with a durable queue.

    Args:
        host: Pika host.
        queue: Pika queue to declare.
        username: Username for authentication with Pika host.
        password: Password for authentication with Pika host.
        port: port of the Pika host.
        connection_attempts: Number of channel attempts before giving up.
        retry_delay_in_seconds: Delay in seconds between channel attempts.

    Returns:
        Pika `BlockingChannel` with declared queue.
    """
    connection = initialise_pika_connection(
        host, username, password, port, connection_attempts, retry_delay_in_seconds
    )

    return _declare_pika_channel_with_queue(connection, queue)


def _declare_pika_channel_with_queue(
    connection: "BlockingConnection", queue: Text
) -> "BlockingChannel":
    """Declare a durable queue on Pika channel."""
    channel = connection.channel()
    channel.queue_declare(queue, durable=True)

    return channel


def close_pika_channel(
    channel: "Channel",
    attempts: int = 1000,
    time_between_attempts_in_seconds: float = 0.001,
) -> None:
    """Attempt to close Pika channel and wait until it is closed.

    Args:
        channel: Pika `Channel` to close.
        attempts: How many times to try to confirm that the channel has indeed been
            closed.
        time_between_attempts_in_seconds: Wait time between attempts to confirm closed
            state.
    """
    from pika.exceptions import AMQPError

    try:
        channel.close()
        logger.debug("Successfully initiated closing of Pika channel.")
    except AMQPError:
        logger.exception("Failed to initiate closing of Pika channel.")

    while attempts:
        if channel.is_closed:
            logger.debug("Successfully closed Pika channel.")
            return None

        time.sleep(time_between_attempts_in_seconds)
        attempts -= 1

    logger.exception("Failed to close Pika channel.")


def close_pika_connection(connection: "Connection") -> None:
    """Attempt to close Pika connection."""
    from pika.exceptions import AMQPError

    try:
        connection.close()
        logger.debug("Successfully closed Pika connection with host.")
    except AMQPError:
        logger.exception("Failed to close Pika connection with host.")


MessageHeaders = Optional[Dict[Text, Text]]
Message = Tuple[Text, MessageHeaders]


class PikaMessageProcessor:
    """A class that holds all the Pika connection details and processes Pika messages."""

    def __init__(
        self,
        parameters: "Parameters",
        queues: Union[List[Text], Tuple[Text], Text, None],
        **kwargs: Any,
    ) -> None:
        """Initialise Pika connector.

        Args:
            parameters: Pika connection parameters
            queues: Pika queues to declare and publish to
        """
        self.parameters: "Parameters" = parameters
        self.queues: List[Text] = self._get_queues_from_args(queues, kwargs)

        self._connection: Optional["SelectConnection"] = None
        self._channel: Optional["Channel"] = None
        self._process_queue: Optional["multiprocessing.Queue"] = None
        self._closing = False

    def __del__(self) -> None:
        if self._channel:
            logger.warning("Closing connection...")
            close_pika_channel(self._channel)
            close_pika_connection(self._channel.connection)

        if self.is_connected:
            self._connection.close()

    def close(self) -> None:
        """Close the Pika connection."""
        self.__del__()

    @staticmethod
    def _get_queues_from_args(
        queues_arg: Union[List[Text], Tuple[Text], Text, None], kwargs: Any
    ) -> Union[List[Text], Tuple[Text]]:
        """Get queues for this event broker.
        The preferred argument defining the RabbitMQ queues the `PikaEventBroker` should
        publish to is `queues` (as of Rasa Open Source version 1.8.2). This function
        ensures backwards compatibility with the old `queue` argument. This method
        can be removed in the future, and `self.queues` should just receive the value of
        the `queues` kwarg in the constructor.
        Args:
            queues_arg: Value of the supplied `queues` argument.
            kwargs: Additional kwargs supplied to the `PikaEventBroker` constructor.
                If `queues_arg` is not supplied, the `queue` kwarg will be used instead.
        Returns:
            Queues this event broker publishes to.
        Raises:
            `ValueError` if no valid `queue` or `queues` argument was found.
        """
        queue_arg = kwargs.pop("queue", None)

        if queue_arg:
            raise_warning(
                "Your Pika event broker config contains the deprecated `queue` key. "
                "Please use the `queues` key instead.",
                FutureWarning,
                docs=DOCS_URL_PIKA_EVENT_BROKER,
            )

        if queues_arg and isinstance(queues_arg, (list, tuple)):
            return list(queues_arg)

        if queues_arg and isinstance(queues_arg, str):
            logger.debug(
                f"Found a string value under the `queues` key of the Pika event broker "
                f"config. Please supply a list of queues under this key, even if it is "
                f"just a single one. See {DOCS_URL_PIKA_EVENT_BROKER}"
            )
            return [queues_arg]

        if queue_arg and isinstance(queue_arg, str):
            return [queue_arg]

        if queue_arg:
            return queue_arg  # pytype: disable=bad-return-type

        raise_warning(
            f"No `queues` or `queue` argument provided. It is suggested to "
            f"explicitly specify a queue as described in "
            f"{DOCS_URL_PIKA_EVENT_BROKER}. "
            f"Using the default queue '{DEFAULT_QUEUE_NAME}' for now."
        )

        return [DEFAULT_QUEUE_NAME]

    @staticmethod
    def _get_message_properties(headers: MessageHeaders = None) -> "BasicProperties":
        """Create RabbitMQ message `BasicProperties`.

        The `app_id` property is set to the value of `RASA_ENVIRONMENT` env variable
        if present, and the message delivery mode is set to 2 (persistent).
        In addition, the `headers` property is set if supplied.

        Args:
            headers: Message headers to add to the message properties of the
                published message (key-value dictionary). The headers can be retrieved
                in the consumer from the `headers` attribute of the message's
                `BasicProperties`.

        Returns:
            `pika.spec.BasicProperties` with the `RASA_ENVIRONMENT` environment variable
            as the properties' `app_id` value, `delivery_mode=2` and `headers` as the
            properties' headers.
        """
        from pika.spec import BasicProperties

        # make message persistent
        kwargs = {"delivery_mode": 2}

        env = os.environ.get("RASA_ENVIRONMENT")
        if env:
            kwargs["app_id"] = env

        if headers:
            kwargs["headers"] = headers

        return BasicProperties(**kwargs)

    @property
    def is_connected(self) -> bool:
        """Indicates if Pika is connected and the channel is initialized.

        Returns:
            A boolean value indicating if the connection is established.
        """
        return self._connection and self._channel

    def is_ready(
        self, attempts: int = 1000, wait_time_between_attempts_in_seconds: float = 0.01
    ) -> bool:
        """Spin until the connector is ready to process messages.

        It typically takes 50 ms or so for the pika channel to open. We'll wait up
        to 10 seconds just in case.

        Args:
            attempts: Number of retries.
            wait_time_between_attempts_in_seconds: Wait time between retries.

        Returns:
            `True` if the channel is available, `False` otherwise.
        """
        while attempts:
            if self.is_connected:
                return True
            time.sleep(wait_time_between_attempts_in_seconds)
            attempts -= 1

        return False

    def _connect(self) -> None:
        """Establish a connection to Pika."""
        self._connection = initialise_pika_select_connection(
            self.parameters, self._on_open_connection, self._on_open_connection_error
        )
        self._run_pika_io_loop_in_thread()

    def _on_open_connection(self, connection: "SelectConnection") -> None:
        logger.debug(
            f"RabbitMQ connection to '{self.parameters.host}' was established."
        )
        connection.add_on_close_callback(self._on_connection_closed)
        connection.channel(on_open_callback=self._on_channel_open)

    def _on_open_connection_error(self, _, error: Text) -> None:
        logger.warning(
            f"Connecting to '{self.parameters.host}' failed with error '{error}'. Trying again."
        )

    def _on_connection_closed(self, _, reason: Any):
        self._channel = None
        if self._closing:
            logger.warning("Connection closing")
            # noinspection PyUnresolvedReferences
            self._connection.ioloop.stop()
        else:
            logger.warning(f"Connection closed, reopening in 5 seconds: {reason}")
            # noinspection PyUnresolvedReferences
            self._connection.ioloop.call_later(5, self._reconnect)

    def _reconnect(self):
        # noinspection PyUnresolvedReferences
        self._connection.ioloop.stop()

        if not self._closing:
            self._connect()

    def _on_channel_open(self, channel: "Channel") -> None:
        logger.debug("RabbitMQ channel was opened. Declaring fanout exchange.")

        channel.add_on_close_callback(self._on_channel_closed)

        # declare exchange of type 'fanout' in order to publish to multiple queues
        # (https://www.rabbitmq.com/tutorials/amqp-concepts.html#exchange-fanout)
        channel.exchange_declare(RABBITMQ_EXCHANGE, exchange_type="fanout")

        for queue in self.queues:
            channel.queue_declare(queue=queue, durable=True)
            channel.queue_bind(exchange=RABBITMQ_EXCHANGE, queue=queue)

        self._channel = channel

    def _on_channel_closed(self, channel: "Channel", reason: Any):
        logger.warning(f"Channel {channel} was closed: {reason}")
        self._connection.close()

    def _publish(self, message: Message) -> None:
        body, headers = message

        self._channel.basic_publish(
            exchange=RABBITMQ_EXCHANGE,
            routing_key="",
            body=body.encode(DEFAULT_ENCODING),
            properties=self._get_message_properties(headers),
        )

    def _process_messages(self) -> None:
        """Start to process messages."""
        logger.debug("Start processing messages...")

        assert self.is_ready()

        try:
            while True:
                message = self._process_queue.get()
                self._publish(message)
                logger.debug(
                    f"Published Pika events to exchange '{RABBITMQ_EXCHANGE}' on host "
                    f"'{self.parameters.host}':\n{message[0]}"
                )
        except EOFError:
            # Will most likely happen when shutting down Rasa X.
            logger.debug(
                "Pika message queue of worker was closed. Stopping to listen for more "
                "messages on this worker."
            )

    def _run_pika_io_loop_in_thread(self) -> None:
        thread = threading.Thread(target=self._run_pika_io_loop)
        thread.start()

    def _run_pika_io_loop(self) -> None:
        # noinspection PyUnresolvedReferences
        self._connection.ioloop.start()

    def run(self, queue: "multiprocessing.Queue") -> None:
        """Run the message processor by connecting to RabbitMQ and then
        starting the IOLoop to block and allow the SelectConnection to operate.

        This function is blocking and indefinite thus it
        should be started in a separate process.
        """
        self._process_queue = queue
        self._connect()
        self._process_messages()


class PikaEventBroker(EventBroker):
    """Pika-based event broker for publishing messages to RabbitMQ."""

    NUMBER_OF_MP_WORKERS = 1
    MP_CONTEXT = None

    if sys.platform == "darwin" and sys.version_info < (3, 8):
        # On macOS, Python 3.8 has switched the default start method to "spawn". To
        # quote the documentation: "The fork start method should be considered
        # unsafe as it can lead to crashes of the subprocess". Apply this fix when
        # running on macOS on Python <= 3.7.x as well.

        # See:
        # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        MP_CONTEXT = "spawn"

    def __init__(
        self,
        host: Text,
        username: Text,
        password: Text,
        port: Union[int, Text] = 5672,
        queues: Union[List[Text], Tuple[Text], Text, None] = None,
        should_keep_unpublished_messages: bool = True,
        raise_on_failure: bool = False,
        log_level: Union[Text, int] = os.environ.get(
            ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES
        ),
        **kwargs: Any,
    ) -> None:
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
            log_level: Logging level.
        """
        logging.getLogger("pika").setLevel(log_level)

        self.host = host
        self.username = username
        self.password = password
        self.port = port
        self.queues = queues
        self.process: Optional[multiprocessing.Process] = None
        self.should_keep_unpublished_messages = should_keep_unpublished_messages
        self.raise_on_failure = raise_on_failure
        self.pika_message_processor: Optional[PikaMessageProcessor] = None
        self.process_queue: multiprocessing.Queue = self._get_mp_context().Queue()

        self._connect()

    def __del__(self) -> None:
        if self.pika_message_processor:
            self.pika_message_processor.close()

        if self.process and self.process.is_alive():
            self.process.terminate()

    def close(self) -> None:
        """Close the Pika connector."""
        self.__del__()

    @classmethod
    def from_endpoint_config(
        cls, broker_config: Optional["EndpointConfig"]
    ) -> Optional["PikaEventBroker"]:
        """Initialise `PikaEventBroker` from `EndpointConfig`.

        Args:
            broker_config: `EndpointConfig` to read.

        Returns:
            `PikaEventBroker` if `broker_config` was supplied, else `None`.
        """
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def _connect(self) -> None:
        parameters = _get_pika_parameters(
            self.host, self.username, self.password, self.port
        )

        self.pika_message_processor = PikaMessageProcessor(
            parameters, queues=self.queues
        )
        self.process = self._start_pika_process()

    def _get_mp_context(self) -> multiprocessing.context.BaseContext:
        return multiprocessing.get_context(self.MP_CONTEXT)

    def _start_pika_process(self) -> Optional[multiprocessing.Process]:
        if self.pika_message_processor:
            process = multiprocessing.Process(
                target=self.pika_message_processor.run, args=(self.process_queue,)
            )
            process.start()
            return process

        return None

    def is_ready(
        self, attempts: int = 1000, wait_time_between_attempts_in_seconds: float = 0.01
    ) -> bool:
        """Spin until Pika is ready to process messages.

        Args:
            attempts: Number of retries.
            wait_time_between_attempts_in_seconds: Wait time between retries.

        Returns:
            `True` if the process is alive, `False` otherwise.
        """
        while attempts:
            if self.process and self.process.is_alive():
                return True
            time.sleep(wait_time_between_attempts_in_seconds)
            attempts -= 1

        return False

    def publish(
        self,
        event: Dict[Text, Any],
        retries: int = 60,
        retry_delay_in_seconds: int = 5,
        headers: Optional[Dict[Text, Text]] = None,
    ) -> None:
        """Publish `event` into Pika queue.

        Args:
            event: Serialised event to be published.
            retries: Number of retries if publishing fails
            retry_delay_in_seconds: Delay in seconds between retries.
            headers: Message headers to append to the published message (key-value
                dictionary). The headers can be retrieved in the consumer from the
                `headers` attribute of the message's `BasicProperties`.
        """
        if not self.process or not self.process.is_alive():
            logger.error("Event broker process has died. Reconnecting...")
            self._connect()

        body = json.dumps(event)
        self.process_queue.put((body, headers))


class PikaProducer(PikaEventBroker):
    def __init__(
        self,
        host: Text,
        username: Text,
        password: Text,
        port: Union[int, Text] = 5672,
        queues: Union[List[Text], Tuple[Text], Text, None] = ("rasa_core_events",),
        should_keep_unpublished_messages: bool = True,
        raise_on_failure: bool = False,
        log_level: Union[Text, int] = os.environ.get(
            ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES
        ),
        **kwargs: Any,
    ):
        raise_warning(
            "The `PikaProducer` class is deprecated, please inherit "
            "from `PikaEventBroker` instead. `PikaProducer` will be "
            "removed in future Rasa versions.",
            FutureWarning,
            docs=DOCS_URL_PIKA_EVENT_BROKER,
        )
        super(PikaProducer, self).__init__(
            host,
            username,
            password,
            port,
            queues,
            should_keep_unpublished_messages,
            raise_on_failure,
            log_level,
            **kwargs,
        )
