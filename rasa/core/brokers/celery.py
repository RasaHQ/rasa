import logging
from functools import partial
from asyncio import AbstractEventLoop
from typing import Dict, Text, Any, Optional

from celery import Celery

from rasa.core.brokers.broker import EventBroker
from rasa.utils.endpoints import EndpointConfig
from rasa.shared.exceptions import RasaException

logger = logging.getLogger(__name__)


class CeleryEventBroker(EventBroker):
    def __init__(
        self,
        broker_url: Optional[Text] = None,
        task_queue_name: Text = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initializes `CeleryBrokerEvent`."""
        if not broker_url:
            raise RasaException(
                "A broker_url must be provided for Celery event broker to work."
            )

        task_instance: Celery = Celery("rasa_event_broker", broker=broker_url)
        self._task = partial(task_instance.send_task, task_queue_name, **kwargs)

    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
        event_loop: Optional[AbstractEventLoop] = None,
    ) -> Optional["CeleryEventBroker"]:
        """Creates broker. See the parent class for more information."""
        if broker_config is None:
            return None

        return cls(**broker_config.kwargs)

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a dict-formatted Rasa Core event into an event queue."""
        self._task(args=[event])
