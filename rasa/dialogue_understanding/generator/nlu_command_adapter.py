from typing import Dict, Text, Any, Optional, List

import structlog

from rasa.dialogue_understanding.commands import (
    Command,
    StartFlowCommand,
    SetSlotCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.commands.utils import (
    triggerable_pattern_to_command_class,
)
from rasa.dialogue_understanding.generator import CommandGenerator
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.slot_mappings import (
    SlotFillingManager,
    extract_slot_value,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import ENTITIES, INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.log_utils import log_llm

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
        execution_context: ExecutionContext,
    ) -> None:
        super().__init__(config)
        self.config = {**self.get_default_config(), **config}
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "NLUCommandAdapter":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

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
        return cls(config, model_storage, resource, execution_context)

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the NLU command adapter."""
        return self._resource

    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        **kwargs: Any,
    ) -> List[Command]:
        """Creates commands using the predicted intents.

        Args:
            message: The message from the user.
            flows: The flows available to the user.
            tracker: The tracker containing the current state of the conversation.
            **kwargs: Keyword arguments for forward compatibility.

        Returns:
            The commands triggered by NLU.
        """
        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return []

        domain = kwargs.get("domain", None)
        commands = self.convert_nlu_to_commands(message, tracker, flows, domain)

        commands_contain_start_flow = any(
            isinstance(command, StartFlowCommand) for command in commands
        )

        if (
            commands
            and commands_contain_start_flow
            and tracker.has_coexistence_routing_slot
        ):
            # if the nlu command adapter will start a flow and the coexistence feature
            # is used, make sure to set the routing slot
            commands += [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)]

        # TODO:
        #    (May 30th, 2024)
        #    This code within the can be removed once the cleaning process
        #    is applied by default for every instance of the command generator
        #    class.
        #    Ticket: https://rasahq.atlassian.net/browse/ENG-1076
        from rasa.dialogue_understanding.processor.command_processor import (
            clean_up_commands,
        )

        log_llm(
            logger=structlogger,
            log_module="NLUCommandAdapter",
            log_event="nlu_command_adapter.predict_commands.finished",
            commands=commands,
        )

        if commands:
            commands = clean_up_commands(
                commands, tracker, flows, self._execution_context
            )
            log_llm(
                logger=structlogger,
                log_module="NLUCommandAdapter",
                log_event="nlu_command_adapter.clean_commands",
                commands=commands,
            )

        return commands

    @staticmethod
    def convert_nlu_to_commands(
        message: Message,
        tracker: DialogueStateTracker,
        flows: FlowsList,
        domain: Optional[Domain] = None,
    ) -> List[Command]:
        """Converts the predicted intent to a command."""
        if tracker is None or flows.is_empty():
            # cannot do anything if there are no flows or no tracker
            return []

        if not (
            message.get(INTENT)
            or message.get(INTENT, {}).get("name")
            or message.get(ENTITIES)
        ):
            # if the message does not have an intent or entities set
            # no commands can be predicted
            return []

        commands: List[Command] = []

        for flow in flows:
            if flow.nlu_triggers and flow.nlu_triggers.is_triggered(message):
                if flow.is_rasa_default_flow:
                    pattern_command = triggerable_pattern_to_command_class.get(flow.id)
                    if pattern_command:
                        commands.append(pattern_command())
                else:
                    commands.append(StartFlowCommand(flow.id))

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

        set_slot_commands = _issue_set_slot_commands(message, tracker, flows, domain)
        commands.extend(set_slot_commands)

        log_llm(
            logger=structlogger,
            log_module="NLUCommandAdapter",
            log_event="nlu_command_adapter.predict_commands",
            commands=commands,
        )

        return commands


def _issue_set_slot_commands(
    message: Message,
    tracker: DialogueStateTracker,
    flows: FlowsList,
    domain: Optional[Domain] = None,
) -> List[Command]:
    """Issue SetSlotCommand for each slot that can be filled with NLU properties."""
    commands: List[Command] = []
    domain = domain if domain else Domain.empty()
    slot_filling_manager = SlotFillingManager(domain, tracker, message)
    available_slot_names = flows.available_slot_names()

    for _, slot in tracker.slots.items():
        # if a slot is not collected in available flows,
        # it means that it is not a slot that can be filled by CALM,
        # so we skip it
        if slot.name not in available_slot_names:
            structlogger.debug("nlu_command_adapter.skip_slot", slot=slot.name)
            continue

        slot_value, is_extracted = extract_slot_value(slot, slot_filling_manager)
        if is_extracted:
            commands.append(
                SetSlotCommand(
                    name=slot.name,
                    value=slot_value,
                    extractor=SetSlotExtractor.NLU.value,
                )
            )

    return commands
