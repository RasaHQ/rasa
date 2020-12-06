import asyncio
import logging
from asyncio import AbstractEventLoop
from typing import Any, Dict, Text, Optional, Union

import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class EventBroker:
    """Base class for any event broker implementation."""

    @staticmethod
    async def create(
        obj: Union["EventBroker", EndpointConfig, None],
        loop: Optional[AbstractEventLoop] = None,
    ) -> Optional["EventBroker"]:
        """Factory to create an event broker."""
        if isinstance(obj, EventBroker):
            return obj

        return await _create_from_endpoint_config(obj, loop)

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> "EventBroker":
        """Creates an `EventBroker` from the endpoint configuration.

        Args:
            broker_config: The configuration for the broker.
            event_loop: The current event loop or `None`.

        Returns:
            An `EventBroker` object.
        """
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

    async def close(self) -> None:
        """Close the connection to an event broker."""
        # default implementation does nothing
        pass


async def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig], event_loop: Optional[AbstractEventLoop]
) -> Optional["EventBroker"]:
    """Instantiate an event broker based on its configuration."""
    if endpoint_config is None:
        broker = None
    elif endpoint_config.type is None or endpoint_config.type.lower() == "pika":
        from rasa.core.brokers.pika import PikaEventBroker

        # default broker if no type is set
        broker = await PikaEventBroker.from_endpoint_config(endpoint_config, event_loop)
    elif endpoint_config.type.lower() == "sql":
        from rasa.core.brokers.sql import SQLEventBroker

        broker = await SQLEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "file":
        from rasa.core.brokers.file import FileEventBroker

        broker = await FileEventBroker.from_endpoint_config(endpoint_config)
    elif endpoint_config.type.lower() == "kafka":
        from rasa.core.brokers.kafka import KafkaEventBroker

        broker = await KafkaEventBroker.from_endpoint_config(endpoint_config)
    else:
        broker = await _load_from_module_name_in_endpoint_config(endpoint_config)

    if broker:
        logger.debug(f"Instantiated event broker to '{broker.__class__.__name__}'.")
    return broker


async def _load_from_module_name_in_endpoint_config(
    broker_config: EndpointConfig,
) -> Optional["EventBroker"]:
    """Instantiate an event broker based on its class name."""
    try:
        event_broker_class = rasa.shared.utils.common.class_from_module_path(
            broker_config.type
        )
        if not asyncio.iscoroutinefunction(event_broker_class.from_endpoint_config):
            rasa.shared.utils.io.raise_deprecation_warning(
                f"The method "
                f"'{EventBroker.__name__}.{EventBroker.from_endpoint_config.__name__}' "
                f"was changed to be 'async'. Your event broker '{broker_config.type}' "
                f"is currently synchronous. Support for synchronous implementations of "
                f"'{EventBroker.from_endpoint_config.__name__}' will be removed in "
                f"3.0.0."
            )
            return event_broker_class.from_endpoint_config(broker_config)
        return await event_broker_class.from_endpoint_config(broker_config)
    except (AttributeError, ImportError) as e:
        logger.warning(
            f"The `EventBroker` type '{broker_config.type}' could not be found. "
            f"Not using any event broker. Error: {e}"
        )
        return None
