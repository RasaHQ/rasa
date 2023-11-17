from typing import Any, Dict, List, Optional

import structlog

from rasa.dialogue_understanding.commands import Command, StartFlowCommand
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import COMMANDS

structlogger = structlog.get_logger()


class CommandGenerator:
    """A command generator.

    Parses a message and returns a list of commands. The commands are then
    executed and will lead to tracker state modifications and action
    predictions."""

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

            try:
                commands = self.predict_commands(message, startable_flows, tracker)
            except Exception as e:
                if isinstance(e, NotImplementedError):
                    raise e
                structlogger.error("command_generator.predict.error", error=e)
                commands = []
            # Double check commands for guarded flows. Unlikely but the llm could
            # have predicted a command for a flow that is not in the startable
            # flow list supplied in the prompt.
            commands = self._check_commands_against_startable_flows(
                commands, startable_flows
            )
            commands_dicts = [command.as_dict() for command in commands]
            message.set(COMMANDS, commands_dicts, add_to_output=True)
        return messages

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

        if commands and len(checked_commands) != len(commands):
            structlogger.info(
                "command_generator.check_commands_against_startable_flows.startable_commands",
                commands=checked_commands,
            )

        return checked_commands
