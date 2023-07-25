import dataclasses
from typing import List, Optional
from rasa.cdu.commands import Command
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import COMMANDS


class CommandGenerator:
    """A command generator.

    Parses a message and returns a list of commands. The commands are then
    executed and will lead to tracker state modifications and action
    predictions."""

    def process(
        self,
        messages: List[Message],
        tracker: Optional[DialogueStateTracker] = None,
        flows: Optional[FlowsList] = None,
    ) -> List[Message]:
        """Process a list of messages. For each message predict commands.

        The result of the generation is added to the message as a list of
        commands.
        """
        for message in messages:
            commands = self.predict_commands(message, tracker, flows)
            commands_dicts = [dataclasses.asdict(command) for command in commands]
            message.set(COMMANDS, commands_dicts, add_to_output=True)
        return messages

    def predict_commands(
        self,
        message: Message,
        tracker: Optional[DialogueStateTracker] = None,
        flows: Optional[FlowsList] = None,
    ) -> List[Command]:
        """Predict commands for a single message.

        Args:
            message: The message to predict commands for.
            tracker: The tracker containing the conversation history up to now.
            flows: The flows to use for command prediction.

        Returns:
            The predicted commands."""
        raise NotImplementedError()
