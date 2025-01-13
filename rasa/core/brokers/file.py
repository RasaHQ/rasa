import json
import logging
import typing
from asyncio import AbstractEventLoop
from typing import Dict, Optional, Text

from rasa.core.brokers.broker import EventBroker

if typing.TYPE_CHECKING:
    from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class FileEventBroker(EventBroker):
    """Log events to a file in json format.

    There will be one event per line and each event is stored as json.
    """

    DEFAULT_LOG_FILE_NAME = "rasa_event.log"

    def __init__(self, path: Optional[Text] = None) -> None:
        self.path = path or self.DEFAULT_LOG_FILE_NAME
        self.event_logger = self._event_logger()

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: Optional["EndpointConfig"],
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> Optional["FileEventBroker"]:
        """Creates broker. See the parent class for more information."""
        if broker_config is None:
            return None

        # noinspection PyArgumentList
        return cls(**broker_config.kwargs)

    def _event_logger(self) -> logging.Logger:
        """Instantiate the file logger."""
        logger_file = self.path
        # noinspection PyTypeChecker
        query_logger = logging.getLogger("event-logger")
        query_logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logger_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        query_logger.propagate = False
        query_logger.addHandler(handler)

        logger.info(f"Logging events to '{logger_file}'.")

        return query_logger

    def publish(self, event: Dict) -> None:
        """Write event to file."""
        self.event_logger.info(json.dumps(event))
        self.event_logger.handlers[0].flush()
