from __future__ import annotations
from typing import Any, Dict, List, Optional

import structlog

from rasa.dialogue_understanding.coexistence.constants import (
    CALM_ENTRY,
    NLU_ENTRY,
    STICKY,
    NON_STICKY,
)
from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.dialogue_understanding.generator.nlu_command_adapter import (
    NLUCommandAdapter,
)
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import COMMANDS, INTENT, INTENT_NAME_KEY
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COEXISTENCE_ROUTER,
    ],
    is_trainable=False,
)
class IntentBasedRouter(GraphComponent):
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            NLU_ENTRY: None,
            CALM_ENTRY: None,
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ) -> None:
        self.config = {**self.get_default_config(), **config}
        self._model_storage = model_storage
        self._resource = resource
        self.validate_config()

    def validate_config(self) -> None:
        """Validate the config of the router."""
        if not (self._is_nlu_entry_valid() and self._is_calm_entry_valid()):
            raise ValueError(
                "The IntentBasedRouter component needs a proper "
                "description of the capabilities implemented in the DM1 "
                "part of the bot."
            )

    def _is_nlu_entry_valid(self) -> bool:
        """Check if the NLU entry in the config is valid."""
        nlu_entry = self.config.get(NLU_ENTRY, {})
        return (
            isinstance(nlu_entry, dict)
            and nlu_entry.get(STICKY) is not None
            and nlu_entry.get(NON_STICKY) is not None
        )

    def _is_calm_entry_valid(self) -> bool:
        """Check if the CALM entry in the config is valid."""
        calm_entry = self.config.get(CALM_ENTRY, {})
        return isinstance(calm_entry, dict) and calm_entry.get(STICKY) is not None

    def train(self, training_data: TrainingData) -> Resource:
        """Train the intent classifier on a data set."""
        return self._resource

    @classmethod
    def load(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "IntentBasedRouter":
        """Loads trained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> IntentBasedRouter:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource)

    async def process(
        self,
        messages: List[Message],
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        domain: Optional[Domain] = None,
    ) -> List[Message]:
        """Process a list of messages."""
        if tracker is None:
            # cannot do anything if there is no tracker (happens during 'rasa test nlu')
            return messages

        for message in messages:
            commands = await self.predict_commands(message, flows, tracker, domain)
            commands_dicts = [command.as_dict() for command in commands]
            message.set(COMMANDS, commands_dicts, add_to_output=True)

        return messages

    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: DialogueStateTracker,
        domain: Optional[Domain] = None,
    ) -> List[Command]:
        if not tracker.has_coexistence_routing_slot:
            raise InvalidConfigException(
                f"Tried to run the IntentBasedRouter component "
                f"without the slot to track coexistence routing ({ROUTE_TO_CALM_SLOT})."
            )

        route_session_to_calm = tracker.get_slot(ROUTE_TO_CALM_SLOT)
        if route_session_to_calm is None:
            commands = self._generate_command_using_intent(message, flows, tracker)
            structlogger.info(
                "intent_based_router.predicated_commands", commands=commands
            )
            return commands
        elif route_session_to_calm is True:
            # don't set any commands so that a `LLMBasedCommandGenerator` is
            # triggered and can predict the actual commands.
            return []
        else:
            # If the session is assigned to DM1 add a `NoopCommand` to silence
            # the other command generators.
            return [NoopCommand()]

    def _check_intent_part_of_nlu_trigger(
        self, message: Message, tracker: DialogueStateTracker, flows: FlowsList
    ) -> bool:
        """Check if the intent is part of a nlu trigger."""
        commands = NLUCommandAdapter.convert_nlu_to_commands(
            message,
            tracker,
            flows,
        )
        return len(commands) > 0

    def _generate_command_using_intent(
        self, message: Message, flows: FlowsList, tracker: DialogueStateTracker
    ) -> List[Command]:
        intent = message.data.get(INTENT)
        if not intent or intent.get(INTENT_NAME_KEY) is None:
            # If the message does not have a predicted intent, let the session
            # be sticky and routed to DM1. This is done as the customers currently
            # are in the inital phase of migration to CALM. In the future, we can
            # let the next command generator predict the commands.
            return [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]

        if intent[INTENT_NAME_KEY] in self.config[NLU_ENTRY][STICKY]:
            # If the intent is in nlu entry sticky, set the slot to route to DM1.
            return [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]
        elif intent[INTENT_NAME_KEY] in self.config[NLU_ENTRY][NON_STICKY]:
            # If the intent is in nlu entry non-sticky, predict a Noop so that
            # the next command generator is skipped for just this turn.
            return [NoopCommand()]
        elif intent[INTENT_NAME_KEY] in self.config[CALM_ENTRY][STICKY]:
            # If the intent is in calm entry sticky, set the slot to route to CALM.
            return []

        if self._check_intent_part_of_nlu_trigger(message, tracker, flows):
            # If the intent is part of a nlu trigger, set the slot to route to CALM.
            return []
        # If the intent is not present in any of the above, let the session be sticky
        # and routed to DM1.
        return [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)]
