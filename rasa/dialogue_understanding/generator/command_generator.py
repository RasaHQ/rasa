from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text

import structlog

from rasa.dialogue_understanding.commands import Command, StartFlowCommand, ErrorCommand
from rasa.shared.core.flows import FlowsList
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
    predictions."""

    def __init__(self, config: Dict[Text, Any]):
        self.user_input_config = UserInputConfig.from_dict(
            config.get("user_input") or {}
        )

    def process(
        self,
        messages: List[Message],
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Message]:
        """Process a list of messages. For each message predict commands.

        The result of the generation is added to the message as a list of
        commands.

        Args:
            messages: The messages to process.
            tracker: The tracker containing the conversation history up to now.
            flows: The flows to use for command prediction.

        Returns:
        The processed messages (usually this is just one during prediction).
        """
        context_and_slots = self._get_context_and_slots(tracker)
        # flow guard check.
        startable_flows = flows.startable_flows(context_and_slots)

        for message in messages:

            if message.get(COMMANDS):
                # do not overwrite commands if they are already present
                # i.e. another command generator already predicted commands
                continue

            commands = self._evaluate_and_predict(message, startable_flows, tracker)
            # Double check commands for guarded flows. Unlikely but the llm could
            # have predicted a command for a flow that is not in the startable
            # flow list supplied in the prompt.
            commands = self._check_commands_against_startable_flows(
                commands, startable_flows
            )
            commands_dicts = [command.as_dict() for command in commands]
            message.set(COMMANDS, commands_dicts, add_to_output=True)

        return messages

    def _evaluate_and_predict(
        self,
        message: Message,
        startable_flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        """Evaluates the given message for errors and predicts commands if no errors
        are found.

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
            return self.predict_commands(message, startable_flows, tracker)
        except NotImplementedError:
            raise
        except Exception as e:
            structlogger.error("command_generator.predict.error", error=str(e))
            return []

    def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        """Predict commands for a single message.

        Args:
            message: The message to predict commands for.
            flows: The flows to use for command prediction.
            tracker: The tracker containing the conversation history up to now.

        Returns:
        The predicted commands.
        """
        raise NotImplementedError()

    def _get_context_and_slots(
        self,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get the context document for the flow guard check.

        Args:
            tracker: The tracker containing the conversation history up to now.

        Returns:
            The startable flows.
        """
        if not tracker:
            return {}

        # Get current context and slots to prepare document for flow guard check.
        return {
            "context": tracker.stack.current_context(),
            "slots": tracker.current_slot_values(),
        }

    def _check_commands_against_startable_flows(
        self, commands: List[Command], startable_flows: FlowsList
    ) -> List[Command]:
        """Check if the start flow commands are only for startable flows.

        Args:
            commands: The commands to check.
            startable_flows: The flows which have their starting conditions statisfied.

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
        structlogger.info(
            "command_generator.check_commands_against_startable_flows.startable_commands",
            commands=checked_commands,
        )
        return checked_commands

    def evaluate_message(self, message: Message) -> List[Command]:
        """Evaluates the given message

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
