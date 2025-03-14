from typing import List

from rasa.dialogue_understanding.commands.prompt_command import PromptCommand


def commands_as_string(commands: List[PromptCommand], delimiter: str = "\n") -> str:
    return delimiter.join([command.to_dsl() for command in commands])
