import logging
from functools import partial
from asyncio import AbstractEventLoop
from typing import Dict, Text, Callable, Any, Optional, Union

from celery import Celery

import rasa.shared.utils.common
from rasa.core.brokers.broker import EventBroker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class CeleryEventBroker(EventBroker):
    def __init__(
        self,
        broker_url: Optional[Text] = None,
        task_queue_name: Text = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initializes `CeleryBrokerEvent`."""
        self._task = self._get_task(broker_url, task_queue_name)
        self._task_kwargs: Dict[Text, Any] = kwargs

    def _get_task(
        self, broker_url: Optional[Text], task_name: Text
    ) -> Callable[..., None]:
        """Return the task based on the provided broker_url and task_name."""
        if broker_url:
            task_instance: Celery = Celery("rasa_event_broker", broker=broker_url)
            task_callable = partial(task_instance.send_task, task_name)
        else:
            task_instance = rasa.shared.utils.common.class_from_module_path(task_name)
            task_callable = partial(task_instance.apply_async)

        if not task_callable:
            raise ValueError("Task is not defined")

        return task_callable

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
        self._task([event], **self._task_kwargs)
