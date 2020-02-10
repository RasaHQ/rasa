import json
import logging
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
        security_protocol="SASL_PLAINTEXT",
        loglevel=logging.ERROR,
    ) -> None:

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
    def from_endpoint_config(cls, broker_config) -> Optional["KafkaEventBroker"]:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(self, event) -> None:
        self._create_producer()
        self._publish(event)
        self._close()

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
            )

    def _publish(self, event) -> None:
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
