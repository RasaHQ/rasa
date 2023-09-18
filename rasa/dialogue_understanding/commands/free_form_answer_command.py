from dataclasses import dataclass
from rasa.dialogue_understanding.commands import Command


@dataclass
class FreeFormAnswerCommand(Command):
    """A command to indicate a free-form answer by the bot."""

    pass
