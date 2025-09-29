import asyncio
import json
import logging
import os
import threading
import time
from asyncio import AbstractEventLoop
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text, Tuple, Union

import structlog

import rasa.shared.utils.common
from rasa.core.brokers.broker import EventBroker
from rasa.core.constants import KAFKA_SERVICE_NAME
from rasa.core.exceptions import KafkaProducerInitializationError
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProviderInput,
    SupportedServiceType,
    create_iam_credentials_provider,
)
from rasa.shared.core.events import ErrorHandled
from rasa.shared.utils.io import DEFAULT_ENCODING
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from confluent_kafka import KafkaError, Message, Producer

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


class KafkaEventBroker(EventBroker):
    """Kafka event broker."""

    def __init__(
        self,
        url: Union[Text, List[Text], None],
        topic: Text = "rasa_core_events",
        client_id: Optional[Text] = None,
        partition_by_sender: bool = False,
        sasl_username: Optional[Text] = None,
        sasl_password: Optional[Text] = None,
        sasl_mechanism: Optional[Text] = "PLAIN",
        ssl_cafile: Optional[Text] = None,
        ssl_certfile: Optional[Text] = None,
        ssl_keyfile: Optional[Text] = None,
        ssl_check_hostname: bool = False,
        security_protocol: Text = "SASL_PLAINTEXT",
        **kwargs: Any,
    ) -> None:
        """Kafka event broker.

        Args:
            url: 'url[:port]' string (or list of 'url[:port]'
                strings) that the producer should contact to bootstrap initial
                cluster metadata. This does not have to be the full node list.
                It just needs to have at least one broker that will respond to a
                Metadata API Request.
            topic: Topics to subscribe to.
            client_id: A name for this client. This string is passed in each request
                to servers and can be used to identify specific server-side log entries
                that correspond to this client. Also submitted to `GroupCoordinator` for
                logging with respect to producer group administration.
            partition_by_sender: Flag to configure whether messages are partitioned by
                sender_id or not
            sasl_username: Username for plain authentication.
            sasl_password: Password for plain authentication.
            sasl_mechanism: Authentication mechanism when security_protocol is
                configured for SASL_PLAINTEXT or SASL_SSL.
                Valid values are: PLAIN, GSSAPI, OAUTHBEARER, SCRAM-SHA-256,
                SCRAM-SHA-512. Default: `PLAIN`
            ssl_cafile: Optional filename of ca file to use in certificate
                verification.

            ssl_certfile : Optional filename of file in pem format containing
                the client certificate, as well as any ca certificates needed to
                establish the certificate's authenticity.

            ssl_keyfile : Optional filename containing the client private key.

            ssl_check_hostname : Flag to configure whether ssl handshake
                should verify that the certificate matches the broker's hostname.

            security_protocol : Protocol used to communicate with brokers.
                Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
        """
        self.producer: Optional[Producer] = None
        self.url = url
        self.topic = topic
        self.client_id = client_id
        self.partition_by_sender = partition_by_sender
        self.security_protocol = security_protocol.upper()
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.sasl_mechanism = sasl_mechanism
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.queue_size = kwargs.get("queue_size")
        self.ssl_check_hostname = "https" if ssl_check_hostname else None
        self.iam_credentials_provider = create_iam_credentials_provider(
            IAMCredentialsProviderInput(
                service_type=SupportedServiceType.EVENT_BROKER,
                service_name=KAFKA_SERVICE_NAME,
            )
        )

        # PII management attributes
        self.stream_pii = kwargs.get("stream_pii", True)
        self.anonymization_topics = kwargs.get("anonymization_topics", [])

        # Async producer implementation followed from confluent-kafka asyncio example:
        # https://github.com/confluentinc/confluent-kafka-python/blob/master/examples/asyncio_example.py#L88  # noqa: E501
        self._loop = asyncio.get_event_loop()
        self._cancelled = False
        self._poll_thread = threading.Thread(target=self._poll_loop)
        self._poll_thread.start()

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> Optional["KafkaEventBroker"]:
        """Creates broker. See the parent class for more information."""
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(
        self,
        event: Dict[Text, Any],
        retries: int = 60,
        retry_delay_in_seconds: float = 5,
    ) -> None:
        """Publishes events."""
        from confluent_kafka import KafkaError, KafkaException

        if retries == 1:
            retries = 2

        if self.producer is None:
            self.producer = self._create_producer()
            try:
                self._check_kafka_connection()
                logger.debug("Connection to kafka successful.")
            except KafkaException as exc:
                logger.debug(f"Failed to connect to kafka: {exc}")
                return
        while retries:
            try:
                self._publish(event)
                return
            except BufferError as e:
                logger.error(
                    f"Could not publish message to kafka url '{self.url}'. "
                    f"Failed with error: {e}"
                )
                self.producer.poll(1)
                retries -= 1
            except Exception as exc:
                if (
                    isinstance(exc, KafkaException)
                    and exc.args[0].code() == KafkaError.MSG_SIZE_TOO_LARGE
                ):
                    logger.warning(
                        "Message size is too large for the Kafka broker. "
                        "Please check the message.max.bytes configuration. "
                        "Sending error event."
                    )

                    original_event_type = event.get("event", "")
                    sender_id = event.get("sender_id", "")
                    event = ErrorHandled(
                        error_code=KafkaError.MSG_SIZE_TOO_LARGE,
                        metadata={
                            "error_msg": f"Skipping message for event type "
                            f"'{original_event_type}' because "
                            f"of Kafka message size limit: {exc.args[0].str()}.",
                            "error_source": "KafkaEventBroker",
                        },
                    ).as_dict()
                    event.update({"sender_id": sender_id})
                else:
                    logger.error(
                        f"Could not publish message to kafka url '{self.url}'. "
                        f"Failed with error: {exc}"
                    )
                self._retry_publish(event, retries, retry_delay_in_seconds)

        logger.error("Failed to publish Kafka event.")

    def _retry_publish(
        self,
        event: Dict[Text, Any],
        retries: int,
        retry_delay_in_seconds: float,
    ) -> None:
        """Retries publishing if the producer is not connected.

        Args:
            event: The event to publish.
        """
        from confluent_kafka import KafkaException

        try:
            self._check_kafka_connection()
        except KafkaException:
            logger.debug("Connection to kafka lost, reconnecting...")
            self.producer = self._create_producer()
            try:
                self._check_kafka_connection()
                logger.debug("Reconnection to kafka successful")
                self._publish(event)
                return
            except KafkaException:
                pass
        retries -= 1
        time.sleep(retry_delay_in_seconds)

    def _check_kafka_connection(self) -> None:
        """Verifies connection with Kafka.

        Raises:
            KafkaException: if Kafka is disconnected.
        """
        if self.producer is not None:
            structlogger.debug(
                "rasa.core.brokers.kafka.KafkaEventBroker.check_kafka_connection",
            )
            # we have to poll to trigger the oauth_cb if using IAM authentication
            self.producer.poll(0)
            self.producer.list_topics(self.topic, timeout=5)

    def get_aws_iam_token(
        self, oauth_config: Any
    ) -> Tuple[Optional[str], Optional[float]]:
        """Callback function to fetch AWS IAM token for MSK authentication.

        The callback function requires this specific signature to work correctly.

        Args:
            oauth_config: OAuth configuration.

        Returns:
            A tuple of auth token and expiry time in seconds.
        """
        if self.iam_credentials_provider is None:
            return None, None

        temp_credentials = self.iam_credentials_provider.get_temporary_credentials()
        return temp_credentials.auth_token, temp_credentials.expiration

    def _get_kafka_config(self) -> Dict[Text, Any]:
        config = {
            "client.id": self.client_id,
            "bootstrap.servers": self.url,
            "error_cb": kafka_error_callback,
        }
        if self.queue_size:
            config["queue.buffering.max.messages"] = self.queue_size

        if self.security_protocol == "PLAINTEXT":
            authentication_params: Dict[Text, Any] = {
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SASL_PLAINTEXT":
            authentication_params = {
                "sasl.username": self.sasl_username,
                "sasl.password": self.sasl_password,
                "sasl.mechanism": self.sasl_mechanism,
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SSL":
            authentication_params = {
                "ssl.ca.location": self.ssl_cafile,
                "ssl.certificate.location": self.ssl_certfile,
                "ssl.key.location": self.ssl_keyfile,
                "security.protocol": self.security_protocol,
            }
        elif self.security_protocol == "SASL_SSL":
            authentication_params = {
                "ssl.ca.location": self.ssl_cafile,
                "ssl.endpoint.identification.algorithm": self.ssl_check_hostname,
                "security.protocol": self.security_protocol,
                "sasl.mechanism": self.sasl_mechanism,
            }

            if self.iam_credentials_provider is not None:
                authentication_params.update(
                    {
                        "oauth_cb": self.get_aws_iam_token,
                    }
                )
            else:
                authentication_params.update(
                    {
                        "sasl.username": self.sasl_username,
                        "sasl.password": self.sasl_password,
                        "ssl.certificate.location": self.ssl_certfile,
                        "ssl.key.location": self.ssl_keyfile,
                    }
                )
        else:
            raise ValueError(
                f"Cannot initialise `KafkaEventBroker`: "
                f"Invalid `security_protocol` ('{self.security_protocol}')."
            )

        return {**config, **authentication_params}

    def _create_producer(self) -> "Producer":
        import confluent_kafka

        try:
            return confluent_kafka.Producer(self._get_kafka_config())
        except confluent_kafka.KafkaException as e:
            raise KafkaProducerInitializationError(
                f"Cannot initialise `KafkaEventBroker`: {e}"
            )

    def _publish(self, event: Dict[Text, Any]) -> None:
        if self.partition_by_sender:
            partition_key = bytes(event.get("sender_id"), encoding=DEFAULT_ENCODING)
        else:
            partition_key = None

        headers = []
        if self.rasa_environment:
            headers = [
                (
                    "RASA_ENVIRONMENT",
                    bytes(self.rasa_environment, encoding=DEFAULT_ENCODING),
                )
            ]

        reduced_event = rasa.shared.core.events.remove_parse_data(event)
        structlogger.debug(
            "kafka.publish.event",
            event_info="Logging a reduced version of the Kafka event",
            topic=self.topic,
            rasa_event=reduced_event,
            partition_key=partition_key,
            headers=headers,
        )

        serialized_event = json.dumps(event).encode(DEFAULT_ENCODING)

        if self.producer is not None:
            self.producer.produce(
                self.topic,
                value=serialized_event,
                key=partition_key,
                headers=headers,
                on_delivery=delivery_report,
            )

    async def close(self) -> None:
        self._cancelled = True
        self._poll_thread.join()
        if self.producer:
            self.producer.flush()

    @cached_property
    def rasa_environment(self) -> Optional[Text]:
        """Get value of the `RASA_ENVIRONMENT` environment variable."""
        return os.environ.get("RASA_ENVIRONMENT", "RASA_ENVIRONMENT_NOT_SET")

    def _poll_loop(self) -> None:
        """Polls the producer for events.

        Required to trigger the on_delivery callback passed to produce method.
        """
        if self.producer is not None:
            while not self._cancelled:
                self.producer.poll(0.1)


def kafka_error_callback(err: "KafkaError") -> None:
    """Callback for Kafka errors.

    Any exception raised from this callback will be re-raised from the
    triggering flush() call.
    """
    from confluent_kafka import KafkaError, KafkaException

    # handle authentication / connection related issues, likely pointing
    # to a configuration error
    if (
        err.code() == KafkaError._ALL_BROKERS_DOWN
        or err.code() == KafkaError._AUTHENTICATION
        or err.code() == KafkaError._MAX_POLL_EXCEEDED
    ):
        raise KafkaException(err)
    else:
        logger.warning("A KafkaError has been raised.", exc_info=True)


def delivery_report(err: Exception, msg: "Message") -> None:
    """Reports the failure or success of a message delivery.

    Args:
        err (KafkaError): The error that occurred on None on success.
        msg (Message): The message that was produced or failed.
    """
    if err is not None:
        logger.error(f"Delivery failed for User record {msg.key()}: {err}")
        return

    logger.info(
        f"User record {msg.key()} successfully produced to "
        f"{msg.topic()} [{msg.partition()}] at offset {msg.offset()}."
    )
