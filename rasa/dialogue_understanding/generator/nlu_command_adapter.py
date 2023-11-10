from typing import Dict, Text, Any, Optional, List

import structlog

from rasa.dialogue_understanding.commands import Command, StartFlowCommand
from rasa.dialogue_understanding.generator import CommandGenerator
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=False,
)
class NLUCommandAdapter(GraphComponent, CommandGenerator):
    """An NLU-based command generator."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.config = {**self.get_default_config(), **config}
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "NLUCommandAdapter":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "NLUCommandAdapter":
        """Loads trained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the NLU command adapter."""
        return self._resource

    def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        """Creates commands using the predicted intents.

        Args:
            message: The message from the user.
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.

        Returns:
            The commands triggered by NLU.
        """
        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return []

        if not message.get(INTENT) or not message.get(INTENT)["name"]:
            # if the message does not have an intent set,
            # no commands can be predicted
            return []

        commands: List[Command] = []

        for flow in flows:
            if flow.nlu_triggers and flow.nlu_triggers.is_triggered(message):
                commands.append(StartFlowCommand(flow.id))

        structlogger.debug("nlu_command_adapter.predict_commands", commands=commands)

        # there should be just one flow that can be triggered by the predicted intent
        # this is checked when loading the flows
        # however we just doublecheck here and return the first command if there are
        # multiple flows triggered by the intent
        if len(commands) > 1:
            structlogger.warning(
                "nlu_command_adapter.predict_commands",
                messag=f"Two many flows found that are triggered by the "
                f"intent '{message.get(INTENT)['name']}'. Take the first one.",
                commands=commands,
            )
            commands = [commands[0]]

        return commands
