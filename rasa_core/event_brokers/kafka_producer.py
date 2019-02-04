import logging
import json
from kafka import KafkaProducer as ProducerKafka
from rasa_core.broker import EventChannel


logger = logging.getLogger(__name__)


class KafkaProducer(EventChannel):
    @classmethod
    def name(cls):
        return "kafka_producer"

    def __init__(self, host, sasl_plain_username=None,
                 sasl_plain_password=None, ssl_cafile=None,
                 ssl_certfile=None, ssl_keyfile=None,
                 ssl_check_hostname=False,
                 topic='rasa_core_events',
                 security_protocol='SASL_PLAINTEXT',
                 loglevel=logging.ERROR):

        self.host = host
        self.topic = topic
        self.security_protocol = security_protocol
        self.sasl_plain_username = sasl_plain_username
        self.sasl_plain_password = sasl_plain_password
        self.ssl_cafile = ssl_cafile
        self.ssl_certfile = ssl_certfile
        self.ssl_keyfile = ssl_keyfile
        self.ssl_check_hostname = ssl_check_hostname

        logging.getLogger('kafka').setLevel(loglevel)

    def publish(self, event):
        self._create_producer()
        self._publish(event)
        self._close()

    def _create_producer(self):
        if self.security_protocol == 'SASL_PLAINTEXT':
            self.producer = ProducerKafka(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                sasl_plain_username=self.sasl_plain_username,
                sasl_plain_password=self.sasl_plain_password,
                sasl_mechanism='PLAIN',
                security_protocol=self.security_protocol)
        elif self.security_protocol == 'SSL':
            self.producer = ProducerKafka(
                bootstrap_servers=[self.host],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                ssl_cafile=self.ssl_cafile,
                ssl_certfile=self.ssl_certfile,
                ssl_keyfile=self.ssl_keyfile,
                ssl_check_hostname=False,
                security_protocol=self.security_protocol)

    def _publish(self, event):
        self.producer.send(self.topic, event)

    def _close(self):
        self.producer.close()
