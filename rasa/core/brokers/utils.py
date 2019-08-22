import logging
import typing
from typing import Optional

import rasa.utils.common as rasa_utils
from rasa.utils.endpoints import EndpointConfig

if typing.TYPE_CHECKING:
    from rasa.core.brokers.event_channel import EventChannel

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
