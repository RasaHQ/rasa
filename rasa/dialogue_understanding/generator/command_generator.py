from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text

import structlog

from rasa.dialogue_understanding.commands import (
    Command,
    SetSlotCommand,
    StartFlowCommand,
    ErrorCommand,
)
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.shared.core.constants import SlotMappingType
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slot_mappings import SlotFillingManager
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import COMMANDS, TEXT
from rasa.shared.utils.llm import DEFAULT_MAX_USER_INPUT_CHARACTERS
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
)

structlogger = structlog.get_logger()


@dataclass
class UserInputConfig:
    """Configuration class for user input settings."""

    max_characters: int = DEFAULT_MAX_USER_INPUT_CHARACTERS
    """The maximum number of characters allowed in the user input."""

    def __post_init__(self) -> None:
        if self.max_characters is None:
            self.max_characters = DEFAULT_MAX_USER_INPUT_CHARACTERS

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserInputConfig":
        return cls(
            max_characters=data.get("max_characters", DEFAULT_MAX_USER_INPUT_CHARACTERS)
        )


class CommandGenerator:
    """A command generator.

    Parses a message and returns a list of commands. The commands are then
    executed and will lead to tracker state modifications and action
    predictions.
    """

    def __init__(self, config: Dict[Text, Any]):
        self.user_input_config = UserInputConfig.from_dict(
            config.get("user_input") or {}
        )

    async def process(
        self,
        messages: List[Message],
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        domain: Optional[Domain] = None,
    ) -> List[Message]:
        """Process a list of messages. For each message predict commands.

        The result of the generation is added to the message as a list of
        commands.

        Args:
            messages: The messages to process.
            tracker: The tracker containing the conversation history up to now.
            flows: The flows to use for command prediction.
            domain: The domain.

        Returns:
        The processed messages (usually this is just one during prediction).
        """
        # Determines a set of startable flows by evaluating flow guard conditions.
        startable_flows = self.get_startable_flows(flows, tracker)
        # Get the currently active and called flow (if present).
        # If they would be guarded, e.g. if: false, they would not be in the list
        # of startable flows and not available inside the prompt.
        active_flows = self.get_active_flows(flows, tracker)
        available_flows = FlowsList.from_multiple_flows_lists(
            startable_flows, active_flows
        )

        for message in messages:
            if message.get(COMMANDS):
                # do not overwrite commands if they are already present
                # i.e. another command generator already predicted commands
                continue

            commands = await self._evaluate_and_predict(
                message, available_flows, tracker, domain
            )
            # Double check commands for guarded flows. Unlikely but the llm could
            # have predicted a command for a flow that is not in the startable
            # flow list supplied in the prompt.
            commands = self._check_commands_against_startable_flows(
                commands, startable_flows
            )
            commands = self._check_commands_against_slot_mappings(
                commands, tracker, domain
            )
            commands_dicts = [command.as_dict() for command in commands]
            message.set(COMMANDS, commands_dicts, add_to_output=True)

        return messages

    def get_startable_flows(
        self, flows: FlowsList, tracker: Optional[DialogueStateTracker] = None
    ) -> FlowsList:
        """Determines a set of startable flows by evaluating flow guard conditions.

        Args:
            flows: Underlying flows.
            tracker: The tracker containing the conversation history up to now.

        Returns:
            FlowsList: All flows for which the starting conditions are met.
        """
        if tracker is not None:
            # if tracker is not None, evaluate the flow guard conditions with
            # the current state of the tracker
            return tracker.get_startable_flows(flows)

        # else evaluate it without the tracker context
        return flows.get_startable_flows({})

    def get_active_flows(
        self, flows: FlowsList, tracker: Optional[DialogueStateTracker]
    ) -> FlowsList:
        """Retrieve a list of currently active flows.

        Args:
            flows: Underlying flows.
            tracker: The tracker.

        Returns:
            FlowsList: All currently active flows.
        """
        if not tracker:
            return FlowsList([])
        return tracker.get_active_flows(flows)

    async def _evaluate_and_predict(
        self,
        message: Message,
        startable_flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        domain: Optional[Domain] = None,
    ) -> List[Command]:
        """Evaluates message for errors and predicts commands if no errors are found.

        Args:
            message: The message to process.
            tracker: The tracker containing the conversation history up to now.
            startable_flows: The startable flows to use for command prediction.

        Returns:
            Errors or predicted commands
        """
        # evaluate message for errors
        if error_commands := self.evaluate_message(message):
            return error_commands

        # if no errors, try predicting commands
        try:
            return await self.predict_commands(
                message, startable_flows, tracker, domain=domain
            )
        except NotImplementedError:
            raise
        except Exception as e:
            structlogger.error("command_generator.predict.error", error=str(e))
            return []

    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        **kwargs: Any,
    ) -> List[Command]:
        """Predict commands for a single message.

        Args:
            message: The message to predict commands for.
            flows: The flows to use for command prediction.
            tracker: The tracker containing the conversation history up to now.
            **kwargs: Keyword arguments for forward compatibility.
        Returns:
        The predicted commands.
        """
        raise NotImplementedError()

    def _check_commands_against_startable_flows(
        self, commands: List[Command], startable_flows: FlowsList
    ) -> List[Command]:
        """Check if the start flow commands are only for startable flows.

        Args:
            commands: The commands to check.
            startable_flows: The flows which have their starting conditions satisfied.

        Returns:
            The commands that are startable.
        """
        checked_commands = [
            command
            for command in commands
            if not (
                isinstance(command, StartFlowCommand)
                and command.flow not in startable_flows.flow_ids
            )
        ]

        if commands and len(checked_commands) != len(commands):
            structlogger.info(
                "command_generator.check_commands_against_startable_flows.startable_commands",
                commands=checked_commands,
            )

        return checked_commands

    def evaluate_message(self, message: Message) -> List[Command]:
        """Evaluates the given message.

        Args:
            message: The message to evaluate.

        Returns:
            A list of error commands indicating the type of error.
        """
        errors: List[Command]

        if self.check_if_message_is_empty(message):
            # notify the user that the message is empty
            errors = [
                ErrorCommand(error_type=RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY)
            ]
        elif self.check_if_message_exceeds_limit(message):
            # notify the user about message length
            errors = [
                ErrorCommand(
                    error_type=RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
                    info={"max_characters": self.user_input_config.max_characters},
                )
            ]
        else:
            return []

        structlogger.info(
            "command_generator.evaluate_message.error",
            event_info="Invalid message",
            errors=[e.as_dict() for e in errors],
            message=message.get(TEXT),
        )
        return errors

    def check_if_message_exceeds_limit(self, message: Message) -> bool:
        """Checks if the given message exceeds the predefined number of characters."""
        # if limit was a negative number, omit it
        if self.user_input_config.max_characters < 0:
            return False
        return len(message.get(TEXT, "")) > self.user_input_config.max_characters

    def check_if_message_is_empty(self, message: Message) -> bool:
        """Checks if the given message is empty or whitespace-only."""
        return len(message.get(TEXT, "").strip()) == 0

    @staticmethod
    def _check_commands_against_slot_mappings(
        commands: List[Command],
        tracker: DialogueStateTracker,
        domain: Optional[Domain] = None,
    ) -> List[Command]:
        """Check if the LLM-issued slot commands are fillable.

        The LLM-issued slot commands are fillable if the slot
        mappings are satisfied.
        """
        if not domain:
            return commands

        llm_fillable_slot_names = [
            command.name
            for command in commands
            if isinstance(command, SetSlotCommand)
            and command.extractor == SetSlotExtractor.LLM.value
        ]

        if not llm_fillable_slot_names:
            return commands

        llm_fillable_slots = [
            slot for slot in domain.slots if slot.name in llm_fillable_slot_names
        ]

        slot_filling_manager = SlotFillingManager(domain, tracker)
        slots_to_be_removed = []

        structlogger.debug(
            "command_processor.check_commands_against_slot_mappings.active_flow",
            active_flow=tracker.active_flow,
        )

        for slot in llm_fillable_slots:
            should_fill_slot = False
            for mapping in slot.mappings:
                mapping_type = SlotMappingType(mapping.get("type"))

                should_fill_slot = slot_filling_manager.should_fill_slot(
                    slot.name, mapping_type, mapping
                )

                if should_fill_slot:
                    break

            if not should_fill_slot:
                structlogger.debug(
                    "command_processor.check_commands_against_slot_mappings.slot_not_fillable",
                    slot_name=slot.name,
                )
                slots_to_be_removed.append(slot.name)

        if not slots_to_be_removed:
            return commands

        filtered_commands = [
            command
            for command in commands
            if not (
                isinstance(command, SetSlotCommand)
                and command.name in slots_to_be_removed
            )
        ]

        return filtered_commands
