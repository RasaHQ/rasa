from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Text, Tuple

import structlog

from rasa.dialogue_understanding.commands import (
    CannotHandleCommand,
    Command,
    CorrectSlotsCommand,
    ErrorCommand,
    HandleCodeChangeCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.utils import (
    _handle_via_nlu_in_coexistence,
)
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    COMMANDS,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.llm import DEFAULT_MAX_USER_INPUT_CHARACTERS

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
            if _handle_via_nlu_in_coexistence(tracker, message):
                # Skip running the CALM pipeline if the message should
                # be handled by the NLU-based system in a coexistence mode.
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

            # During force slot filling, keep only the command that sets the
            # slot asked by the active collect step.
            # Or return a CannotHandleCommand if no matching command is found.
            commands = self._filter_commands_during_force_slot_filling(
                commands, flows, tracker
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

    def _check_commands_overlap(
        self, prior_commands: List[Command], commands: List[Command]
    ) -> List[Command]:
        """Check if there is overlap between the prior commands and the current ones.

        Args:
            prior_commands: The prior commands.
            commands: The commands to check.

        Returns:
            The final commands.
        """
        if not prior_commands:
            return commands

        prior_commands, commands = self._check_slot_command_overlap(
            prior_commands, commands
        )

        prior_start_flow_names = {
            command.flow
            for command in prior_commands
            if isinstance(command, StartFlowCommand)
        }
        current_start_flow_names = {
            command.flow
            for command in commands
            if isinstance(command, StartFlowCommand)
        }

        return self._check_start_flow_command_overlap(
            prior_commands,
            commands,
            prior_start_flow_names,
            current_start_flow_names,
        )

    def _check_start_flow_command_overlap(
        self,
        prior_commands: List[Command],
        commands: List[Command],
        prior_start_flow_names: Set[str],
        current_start_flow_names: Set[str],
    ) -> List[Command]:
        """Get the final commands.

        Args:
            prior_commands: The prior commands.
            commands: The currently predicted commands to check.
            prior_start_flow_names: The names of the flows from the prior commands.
            current_start_flow_names: The names of the flows from the current commands.

        Returns:
            The final commands.
        """
        raise NotImplementedError()

    def _check_slot_command_overlap(
        self,
        prior_commands: List[Command],
        commands: List[Command],
    ) -> Tuple[List[Command], List[Command]]:
        """Check if the current commands overlap with the prior commands."""
        prior_slot_names = gather_slot_names(prior_commands)
        current_slot_names = gather_slot_names(commands)
        overlapping_slot_names = prior_slot_names.intersection(current_slot_names)

        structlogger.debug(
            "command_generator.check_slot_command_overlap",
            overlapping_slot_names=overlapping_slot_names,
        )

        if not overlapping_slot_names:
            return prior_commands, commands

        return self._filter_slot_commands(
            prior_commands, commands, overlapping_slot_names
        )

    def _filter_slot_commands(
        self,
        prior_commands: List[Command],
        commands: List[Command],
        overlapping_slot_names: Set[str],
    ) -> Tuple[List[Command], List[Command]]:
        """Filter out the overlapping slot commands."""
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
                commands=[
                    checked_command.__class__.__name__
                    for checked_command in checked_commands
                ],
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
    def _get_prior_commands(message: Message) -> List[Command]:
        """Get the prior commands from the tracker."""
        return [
            Command.command_from_json(command) for command in message.get(COMMANDS, [])
        ]

    @staticmethod
    def _filter_commands_during_force_slot_filling(
        commands: List[Command],
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        """Filter commands during a collect step that has set `force_slot_filling`.

        Args:
            commands: The commands to filter.
            flows: All flows.
            tracker: The tracker.

        Returns:
            The filtered commands.
        """
        from rasa.dialogue_understanding.processor.command_processor import (
            find_updated_flows,
            get_current_collect_step,
        )

        if tracker is None:
            structlogger.debug(
                "command_generator.filter_commands_during_force_slot_filling.tracker_not_found",
            )
            return commands

        updated_flows = find_updated_flows(tracker, flows)
        if updated_flows:
            structlogger.debug(
                "command_generator.filter_commands_during_force_slot_filling.running_flows_were_updated",
                updated_flow_ids=updated_flows,
            )
            return [HandleCodeChangeCommand()]

        stack = tracker.stack
        step = get_current_collect_step(stack, flows)

        if step is None or not step.force_slot_filling:
            return commands

        # Retain only the command that sets the slot asked by
        # the active collect step
        filtered_commands: List[Command] = [
            command
            for command in commands
            if (isinstance(command, SetSlotCommand) and command.name == step.collect)
        ]

        if not filtered_commands:
            # If no commands were predicted, we need to return a CannotHandleCommand
            structlogger.debug(
                "command_generator.filter_commands_during_force_slot_filling.no_commands",
                event_info=f"The command generator did not find any SetSlot "
                f"command at the collect step for the slot '{step.collect}'. "
                f"Returning a CannotHandleCommand instead.",
            )
            return [CannotHandleCommand()]

        structlogger.debug(
            "command_generator.filter_commands_during_force_slot_filling.filtered_commands",
            slot_name=step.collect,
            filtered_commands=filtered_commands,
        )

        return filtered_commands


def gather_slot_names(commands: List[Command]) -> Set[str]:
    """Gather all slot names from the commands."""
    slot_names = set()
    for command in commands:
        if isinstance(command, SetSlotCommand):
            slot_names.add(command.name)
        if isinstance(command, CorrectSlotsCommand):
            for slot in command.corrected_slots:
                slot_names.add(slot.name)

    return slot_names
