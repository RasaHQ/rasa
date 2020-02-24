import logging
from typing import Any, Dict, Text, Optional, Union

from rasa.utils import common
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class EventBroker:
    """Base class for any event broker implementation."""

    @staticmethod
    def create(
        obj: Union["EventBroker", EndpointConfig, None],
    ) -> Optional["EventBroker"]:
        """Factory to create an event broker."""

        if isinstance(obj, EventBroker):
            return obj

        return _create_from_endpoint_config(obj)

    @classmethod
    def from_endpoint_config(cls, broker_config: EndpointConfig) -> "EventBroker":
        raise NotImplementedError(
            "Event broker must implement the `from_endpoint_config` method."
        )

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""

        raise NotImplementedError("Event broker must implement the `publish` method.")

    def is_ready(self) -> bool:
        """Determine whether or not the event broker is ready.

        Returns:
            `True` by default, but this may be overridden by subclasses.
        """
        return True

    def close(self) -> None:
        """Close the connection to an event broker."""

        # default implementation does nothing
        pass


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig],
) -> Optional["EventBroker"]:
    """Instantiate an event broker based on its configuration."""

    if endpoint_config is None:
        broker = None
    elif endpoint_config.type is None or endpoint_config.type.lower() == "pika":
        from rasa.core.brokers.pika import PikaEventBroker

        # default broker if no type is set
        broker = PikaEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "sql":
        from rasa.core.brokers.sql import SQLEventBroker

        broker = SQLEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "file":
        from rasa.core.brokers.file import FileEventBroker

        broker = FileEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "kafka":
        from rasa.core.brokers.kafka import KafkaEventBroker

        broker = KafkaEventBroker.from_endpoint_config(endpoint_config)
    else:
        broker = _load_from_module_string(endpoint_config)

    if broker:
        logger.debug(f"Instantiated event broker to '{broker.__class__.__name__}'.")
    return broker


def _load_from_module_string(broker_config: EndpointConfig,) -> Optional["EventBroker"]:
    """Instantiate an event broker based on its class name."""

    try:
        event_broker_class = common.class_from_module_path(broker_config.type)
        return event_broker_class.from_endpoint_config(broker_config)
    except (AttributeError, ImportError) as e:
        logger.warning(
            f"The `EventBroker` type '{broker_config.type}' could not be found. "
            f"Not using any event broker. Error: {e}"
        )
        return None
