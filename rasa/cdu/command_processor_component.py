from __future__ import annotations

from typing import Any, Dict, List, Text
from rasa.cdu.command_processor import execute_commands

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import Event
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


class CommandProcessorComponent(GraphComponent):
    """Processes commands by issuing events to modify a tracker."""

    def __init__(
        self,
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        """Creates flows provider."""
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CommandProcessorComponent:
        """Creates component (see parent class for full docstring)."""
        return cls(model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> CommandProcessorComponent:
        """Creates provider using a persisted version of itself."""
        return cls(model_storage, resource)

    def execute_commands(
        self, tracker: DialogueStateTracker, flows: FlowsList
    ) -> List[Event]:
        """Excute commands in flows to update tracker state."""
        return execute_commands(tracker, flows)
