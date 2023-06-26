from functools import partial
from typing import Dict, Text, Callable, Any

import rasa.shared.utils.common
from rasa.core.brokers.broker import EventBroker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

class CeleryEventBroker(EventBroker):

    
    def __init__(
        self,
        broker_url: Optional[Text] = None,
        task_name: Optional[Text] = None,
        **kwargs,
    ) -> None:
        """Initializes `CeleryBrokerEvent`."""
        
        self._task = self._get_task(broker_url, task_name)
        self._task_kwargs: Dict[Text, Any] = kwargs

    def _get_task(self, broker_url: Optional[Text], task_name: Text) -> Callable[..., None]:
        """Return the task based on the provided broker_url and task_name."""
        if broker_url:
            task_instance = Celery('rasa_event_broker', broker=broker_url)
            task_callable = partial(task_instance.send_task, task_name)
        else:
            task_instance = rasa.shared.utils.common.class_from_module_path(task_name)
            task_callable = partial(task_instance.apply_async) 
        
        return task_callable


    @classmethod
    async def from_endpoint_config(
        cls,
        broker_config: EndpointConfig,
    ) -> "CeleryEventBroker":
        """Creates broker. See the parent class for more information."""
        if broker_config is None:
            return None

        return cls(**broker_config.kwargs)

    def publish(self, event: Dict[Text, Any]) -> None:
        """Publishes a dict-formatted Rasa Core event into an event queue."""
        if not self._task:
            raise ValueError("Task is not defined")

        self._task([event], **self._task_kwargs)
        
        