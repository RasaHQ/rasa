import json
import logging
from asyncio import AbstractEventLoop
from typing import Any, Text, List, Optional, Union, Dict

from rasa.core.brokers.broker import EventBroker
from rasa.shared.utils.io import DEFAULT_ENCODING
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class KafkaEventBroker(EventBroker):
    def __init__(
        self,
        url: Union[Text, List[Text], None],
        topic: Text = "rasa_core_events",
        client_id: Optional[Text] = None,
        sasl_username: Optional[Text] = None,
        sasl_password: Optional[Text] = None,
        ssl_cafile: Optional[Text] = None,
        ssl_certfile: Optional[Text] = None,
        ssl_keyfile: Optional[Text] = None,
        ssl_check_hostname: bool = False,
        security_protocol: Text = "SASL_PLAINTEXT",
        loglevel: Union[int, Text] = logging.ERROR,
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
            group_id: The name of the producer group to join for dynamic partition
                assignment (if enabled), and to use for fetching and committing offsets.
                If None, auto-partition assignment (via group coordinator) and offset
                commits are disabled.
            sasl_username: Username for plain authentication.
            sasl_password: Password for plain authentication.
            ssl_cafile: Optional filename of ca file to use in certificate
                verification.
            ssl_certfile: Optional filename of file in pem format containing
                the client certificate, as well as any ca certificates needed to
                establish the certificate's authenticity.
            ssl_keyfile: Optional filename containing the client private key.
            ssl_check_hostname: Flag to configure whether ssl handshake
                should verify that the certificate matches the brokers hostname.
            security_protocol: Protocol used to communicate with brokers.
                Valid values are: PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL.
            loglevel: Logging level of the kafka logger.

        """
        import kafka

        self.producer = None
        self.url = url
        self.topic = topic
        self.client_id = client_id
        self.security_protocol = security_protocol.upper()
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_check_hostname = ssl_check_hostname

        self.producer: Optional[kafka.KafkaConsumer] = None

        logging.getLogger("kafka").setLevel(loglevel)

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

    def publish(self, event) -> None:
        self._create_producer()
        self._publish(event)
        self._close()

    def _create_producer(self) -> None:
        import kafka

        if self.security_protocol == "PLAINTEXT":
            self.producer = kafka.KafkaProducer(
                client_id=self.client_id,
                bootstrap_servers=self.url,
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                security_protocol=self.security_protocol,
                ssl_check_hostname=False,
            )
        elif self.security_protocol == "SASL_PLAINTEXT":
            self.producer = kafka.KafkaProducer(
                client_id=self.client_id,
                bootstrap_servers=self.url,
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                sasl_plain_username=self.sasl_username,
                sasl_plain_password=self.sasl_password,
                sasl_mechanism="PLAIN",
                security_protocol=self.security_protocol,
            )
        elif self.security_protocol == "SSL":
            self.producer = kafka.KafkaProducer(
                client_id=self.client_id,
                bootstrap_servers=self.url,
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                ssl_cafile=self.ssl_cafile,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
                ssl_check_hostname=False,
                security_protocol=self.security_protocol,
            )
        elif self.security_protocol == "SASL_SSL":
            self.producer = kafka.KafkaProducer(
                client_id=self.client_id,
                bootstrap_servers=self.url,
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                sasl_plain_username=self.sasl_username,
                sasl_plain_password=self.sasl_password,
                ssl_cafile=self.ssl_cafile,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
                ssl_check_hostname=self.ssl_check_hostname,
                security_protocol=self.security_protocol,
                sasl_mechanism="PLAIN",
            )
        else:
            raise ValueError(
                f"Cannot initialise `KafkaEventBroker`: "
                f"Invalid `security_protocol` ('{self.security_protocol}')."
            )

    def _publish(self, event) -> None:
        self.producer.send(self.topic, event)

    def _close(self) -> None:
        self.producer.close()
