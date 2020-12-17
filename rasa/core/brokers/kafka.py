import json
import logging
import time
from typing import Optional

from rasa.constants import DOCS_URL_EVENT_BROKERS
from rasa.core.brokers.broker import EventBroker
from rasa.utils.common import raise_warning
from rasa.utils.io import DEFAULT_ENCODING

logger = logging.getLogger(__name__)


class KafkaEventBroker(EventBroker):
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
        client_id=None,
        group_id=None,
        security_protocol="SASL_PLAINTEXT",
        loglevel=logging.ERROR,
    ) -> None:

        self.producer = None
        self.host = host
        self.topic = topic
        self.client_id = client_id
        self.group_id = group_id
        self.security_protocol = security_protocol
        self.sasl_username = sasl_username
        self.sasl_password = sasl_password
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_check_hostname = ssl_check_hostname

        logging.getLogger("kafka").setLevel(loglevel)

    @classmethod
    def from_endpoint_config(cls, broker_config) -> Optional["KafkaEventBroker"]:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(self, event, retries=60, retry_delay_in_seconds=5) -> None:
        if self.producer is None:
            self._create_producer()
            connected = self.producer.bootstrap_connected()
            if connected:
                logger.debug("Connection to kafka successful.")
            else:
                logger.debug("Failed to connect kafka.")
                return
        while retries:
            try:
                self._publish(event)
                return
            except Exception as e:
                logger.error(
                    f"Could not publish message to kafka host '{self.host}'. "
                    f"Failed with error: {e}"
                )
                connected = self.producer.bootstrap_connected()
                if not connected:
                    self._close()
                    logger.debug("Connection to kafka lost, reconnecting...")
                    self._create_producer()
                    connected = self.producer.bootstrap_connected()
                    if connected:
                        logger.debug("Reconnection to kafka successful")
                        self._publish(event)
                retries -= 1
                time.sleep(retry_delay_in_seconds)

        logger.error("Failed to publish Kafka event.")

    def _create_producer(self) -> None:
        import kafka

        if self.security_protocol == "SASL_PLAINTEXT":
            self.producer = kafka.KafkaProducer(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                sasl_plain_username=self.sasl_username,
                sasl_plain_password=self.sasl_password,
                sasl_mechanism="PLAIN",
                security_protocol=self.security_protocol,
                client_id=self.client_id,
                group_id=self.group_id,
            )
        elif self.security_protocol == "PLAINTEXT":
            self.producer = kafka.KafkaProducer(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                security_protocol=self.security_protocol,
                client_id=self.client_id,
                group_id=self.group_id,
            )
        elif self.security_protocol == "SSL":
            self.producer = kafka.KafkaProducer(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                ssl_cafile=self.ssl_cafile,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
                ssl_check_hostname=False,
                security_protocol=self.security_protocol,
                client_id=self.client_id,
                group_id=self.group_id,
            )
        elif self.security_protocol == "SASL_SSL":
            self.producer = kafka.KafkaProducer(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode(DEFAULT_ENCODING),
                ssl_cafile=self.ssl_cafile,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
                ssl_check_hostname=False,
                sasl_plain_username=self.sasl_username,
                sasl_plain_password=self.sasl_password,
                sasl_mechanism="PLAIN",
                security_protocol=self.security_protocol,
                client_id=self.client_id,
                group_id=self.group_id,
            )
        else:
            logger.error("Kafka security_protocol invalid or not set")

    def _publish(self, event) -> None:
        logger.debug(f"Calling kafka send({self.topic}, {event})")
        self.producer.send(self.topic, event)

    def _close(self) -> None:
        self.producer.close()


class KafkaProducer(KafkaEventBroker):
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
    ) -> None:
        raise_warning(
            "The `KafkaProducer` class is deprecated, please inherit "
            "from `KafkaEventBroker` instead. `KafkaProducer` will be "
            "removed in future Rasa versions.",
            FutureWarning,
            docs=DOCS_URL_EVENT_BROKERS,
        )

        super(KafkaProducer, self).__init__(
            host,
            sasl_username,
            sasl_password,
            ssl_cafile,
            ssl_certfile,
            ssl_keyfile,
            ssl_check_hostname,
            topic,
            security_protocol,
            loglevel,
        )
