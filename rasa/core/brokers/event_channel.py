import logging
from typing import Any, Dict, Text, Optional

from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class EventChannel(object):
    @classmethod
    def from_endpoint_config(cls, broker_config: EndpointConfig) -> "EventChannel":
        raise NotImplementedError(
            "Event broker must implement the `from_endpoint_config` method."
        )

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a json-formatted Rasa Core event into an event queue."""

        raise NotImplementedError("Event broker must implement the `publish` method.")
