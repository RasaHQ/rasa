from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import pika
from typing import Text

logger = logging.getLogger(__name__)


class EventChannel(object):
    def publish(self, event):
        # type: (Text) -> None
        """Publishes a json-formatted Rasa Core event into an event queue."""

        raise NotImplementedError("Event broker must implement the `publish` "
                                  "method")


class PikaProducer(EventChannel):
    def __init__(self, host, username, password,
                 queue='rasa_core_events',
                 loglevel=logging.INFO):

        logging.getLogger('pika').setLevel(loglevel)

        self.queue = queue
        self.host = host
        self.credentials = pika.PlainCredentials(username, password)

    def publish(self, event):
        self._open_connection()
        self._publish(event)
        self._close()

    def _open_connection(self):
        parameters = pika.ConnectionParameters(self.host,
                                               credentials=self.credentials,
                                               connection_attempts=20,
                                               retry_delay=5)
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        self.channel.queue_declare(self.queue, durable=True)

    def _publish(self, body):
        self.channel.basic_publish('', self.queue, body)
        logger.debug('Published pika events to queue {} at '
                     '{}:\n{}'.format(self.queue, self.host, body))

    def _close(self):
        self.connection.close()
