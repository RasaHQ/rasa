from __future__ import annotations

from typing import Any, Dict, List, Optional, Text

import rasa.dialogue_understanding.processor.command_processor
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.structures import StoryGraph


class CommandProcessorComponent(GraphComponent):
    """Processes commands by issuing events to modify a tracker.

    Minimal component that applies commands to a tracker.
    """

    def __init__(self, execution_context: ExecutionContext):
        self._execution_context = execution_context

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CommandProcessorComponent:
        """Creates component (see parent class for full docstring)."""
        return cls(execution_context)

    def execute_commands(
        self,
        tracker: DialogueStateTracker,
        flows: FlowsList,
        domain: Optional[Domain] = None,
        story_graph: Optional[StoryGraph] = None,
    ) -> List[Event]:
        """Execute commands to update tracker state."""
        return rasa.dialogue_understanding.processor.command_processor.execute_commands(
            tracker, flows, self._execution_context, story_graph, domain
        )
