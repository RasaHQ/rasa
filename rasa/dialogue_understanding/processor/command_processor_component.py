from __future__ import annotations

from typing import Any, Dict, List, Text
from rasa.dialogue_understanding.processor.command_processor import execute_commands

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import Event
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


class CommandProcessorComponent(GraphComponent):
    """Processes commands by issuing events to modify a tracker.

    Minimal component that applies commands to a tracker."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CommandProcessorComponent:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def execute_commands(
        self, tracker: DialogueStateTracker, flows: FlowsList
    ) -> List[Event]:
        """Excute commands to update tracker state."""
        return execute_commands(tracker, flows)
