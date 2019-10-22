import logging
import os
import typing
from typing import Optional, Text

import rasa.utils.common as rasa_utils
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa.core.brokers.event_channel import EventChannel
    import pika

logger = logging.getLogger(__name__)


def from_endpoint_config(
    broker_config: Optional[EndpointConfig]
) -> Optional["EventChannel"]:
    """Instantiate an event channel based on its configuration."""

    if broker_config is None:
        return None
    elif broker_config.type == "pika" or broker_config.type is None:
        from rasa.core.brokers.pika import PikaProducer

        return PikaProducer.from_endpoint_config(broker_config)
    elif broker_config.type.lower() == "sql":
        from rasa.core.brokers.sql import SQLProducer

        return SQLProducer.from_endpoint_config(broker_config)
    elif broker_config.type == "file":
        from rasa.core.brokers.file_producer import FileProducer

        return FileProducer.from_endpoint_config(broker_config)
    elif broker_config.type == "kafka":
        from rasa.core.brokers.kafka import KafkaProducer

        return KafkaProducer.from_endpoint_config(broker_config)
    else:
        return load_event_channel_from_module_string(broker_config)


def load_event_channel_from_module_string(
    broker_config: EndpointConfig
) -> Optional["EventChannel"]:
    """Instantiate an event channel based on its class name."""

    try:
        event_channel = rasa_utils.class_from_module_path(broker_config.type)
        return event_channel.from_endpoint_config(broker_config)
    except (AttributeError, ImportError) as e:
        logger.warning(
            "EventChannel type '{}' not found. "
            "Not using any event channel. Error: {}".format(broker_config.type, e)
        )
        return None


def create_rabbitmq_ssl_options(
    rabbitmq_host: Optional[Text] = None
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

        logger.debug(
            "Configuring SSL context for RabbitMQ host '{}'.".format(rabbitmq_host)
        )

        ca_file_path = os.environ.get("RABBITMQ_SSL_CA_FILE")
        key_password = os.environ.get("RABBITMQ_SSL_KEY_PASSWORD")

        ssl_context = rasa.server.create_ssl_context(
            client_certificate_path, client_key_path, ca_file_path, key_password
        )
        return pika.SSLOptions(ssl_context, rabbitmq_host)
    else:
        return None
