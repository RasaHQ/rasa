from typing import List, Optional
import structlog
from rasa.dialogue_understanding.commands import Command
from rasa.shared.core.flows.flow import FlowsList
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
        for message in messages:
            try:
                commands = self.predict_commands(message, flows, tracker)
            except Exception as e:
                if isinstance(e, NotImplementedError):
                    raise e
                structlogger.error("command_generator.predict.error", error=e)
                commands = []
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
