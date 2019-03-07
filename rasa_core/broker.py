import json
import logging
from typing import Any, Dict, Text, Optional

import pika
from rasa_core.utils import class_from_module_path, EndpointConfig

logger = logging.getLogger(__name__)


def from_endpoint_config(broker_config: Optional[EndpointConfig]
                         ) -> Optional['EventChannel']:
    """Instantiate an event channel based on its configuration."""

    if broker_config is None:
        return None
    elif broker_config.type == 'pika' or broker_config.type is None:
        return PikaProducer.from_endpoint_config(broker_config)
    elif broker_config.type == 'file':
        return FileProducer.from_endpoint_config(broker_config)
    else:
        return load_event_channel_from_module_string(broker_config)


def load_event_channel_from_module_string(broker_config: EndpointConfig
                                          ) -> Optional['EventChannel']:
    """Instantiate an event channel based on its class name."""

    try:
        event_channel = class_from_module_path(broker_config.type)
        return event_channel.from_endpoint_config(broker_config)
    except (AttributeError, ImportError) as e:
        logger.warning("EventChannel type '{}' not found. "
                       "Not using any event channel. Error: {}"
                       .format(broker_config.type, e))
        return None


class EventChannel(object):
    @classmethod
    def from_endpoint_config(cls, broker_config: EndpointConfig
                             ) -> 'EventChannel':
        raise NotImplementedError("Event broker must implement the "
                                  "`from_endpoint_config` method.")

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""

        raise NotImplementedError("Event broker must implement the `publish` "
                                  "method.")


class PikaProducer(EventChannel):
    def __init__(self, host, username, password,
                 queue='rasa_core_events',
                 loglevel=logging.INFO):

        logging.getLogger('pika').setLevel(loglevel)

        self.queue = queue
        self.host = host
        self.credentials = pika.PlainCredentials(username, password)

    @classmethod
    def from_endpoint_config(cls, broker_config) -> Optional['PikaProducer']:
        if broker_config is None:
            return None

        return cls(broker_config.url, **broker_config.kwargs)

    def publish(self, event):
        self._open_connection()
        self._publish(json.dumps(event))
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


class FileProducer(EventChannel):
    """Log events to a file in json format.

    There will be one event per line and each event is stored as json."""

    DEFAULT_LOG_FILE_NAME = "rasa_event.log"

    def __init__(self, path: Optional[Text] = None) -> None:
        self.path = path or self.DEFAULT_LOG_FILE_NAME
        self.event_logger = self._event_logger()

    @classmethod
    def from_endpoint_config(cls, broker_config: Optional['EndpointConfig']
                             ) -> Optional['FileProducer']:
        if broker_config is None:
            return None

        # noinspection PyArgumentList
        return cls(**broker_config.kwargs)

    def _event_logger(self):
        """Instantiate the file logger."""

        logger_file = self.path
        # noinspection PyTypeChecker
        query_logger = logging.getLogger('event-logger')
        query_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logger_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        query_logger.propagate = False
        query_logger.addHandler(handler)

        logger.info("Logging events to '{}'.".format(logger_file))

        return query_logger

    def publish(self, event: Dict) -> None:
        """Write event to file."""

        self.event_logger.info(json.dumps(event))
        self.event_logger.handlers[0].flush()
