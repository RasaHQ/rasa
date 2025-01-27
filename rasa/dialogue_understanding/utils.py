from contextlib import contextmanager
from typing import Generator, List, Optional

from rasa.dialogue_understanding.commands import Command
from rasa.shared.nlu.constants import (
    KEY_SYSTEM_PROMPT,
    KEY_USER_PROMPT,
    PREDICTED_COMMANDS,
    PROMPTS,
)
from rasa.shared.nlu.training_data.message import Message

record_commands_and_prompts = False


@contextmanager
def set_record_commands_and_prompts() -> Generator:
    global record_commands_and_prompts
    record_commands_and_prompts = True
    try:
        yield
    finally:
        record_commands_and_prompts = False


def add_commands_to_message_parse_data(
    message: Message, component_name: str, commands: List[Command]
) -> None:
    """Add commands to the message parse data.

    Commands are only added in case the flag 'record_commands_and_prompts' is set.
    Example of predicted commands in the message parse data:
        Message(data={
          PREDICTED_COMMANDS: {
            "MultiStepLLMCommandGenerator": [
                {"command": "set_slot", "name": "slot_name", "value": "slot_value"},
            ],
            "NLUCommandAdapter": [
                {"command": "start_flow", "name": "test_flow"},
            ]
          }
        })
    """
    # only set commands if the flag "record_commands_and_prompts" is set to True
    if not record_commands_and_prompts:
        return

    commands_as_dict = [command.as_dict() for command in commands]

    if message.get(PREDICTED_COMMANDS) is not None:
        predicted_commands = message.get(PREDICTED_COMMANDS)
        if component_name in predicted_commands:
            predicted_commands[component_name].extend(commands_as_dict)
        else:
            predicted_commands[component_name] = commands_as_dict
    else:
        predicted_commands = {component_name: commands_as_dict}

    message.set(
        PREDICTED_COMMANDS,
        predicted_commands,
        add_to_output=True,
    )


def add_prompt_to_message_parse_data(
    message: Message,
    component_name: str,
    prompt_name: str,
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> None:
    """Add prompt to the message parse data.

    Prompt is only added in case the flag 'record_commands_and_prompts' is set.
    Example of prompts in the message parse data:
        Message(data={
          PROMPTS: {
            "MultiStepLLMCommandGenerator": [
                (
                    "fill_slots_prompt",
                    {
                        "user_prompt": <prompt content>",
                        "system_prompt": <prompt content>"
                    }
                ),
                (
                    "handle_flows_prompt",
                    {
                        "user_prompt": <prompt content>",
                        "system_prompt": <prompt content>"
                    }
                ),
            ],
            "SingleStepLLMCommandGenerator": [
                (
                    "prompt_template",
                    {
                        "user_prompt": <prompt content>",
                        "system_prompt": <prompt content>"
                    }
                ),
            ]
          }
        })
    """
    # only set prompt if the flag "record_commands_and_prompts" is set to True
    if not record_commands_and_prompts:
        return

    prompt_tuple = (
        prompt_name,
        {
            KEY_USER_PROMPT: user_prompt,
            **({KEY_SYSTEM_PROMPT: system_prompt} if system_prompt else {}),
        },
    )

    if message.get(PROMPTS) is not None:
        prompts = message.get(PROMPTS)
        if component_name in prompts:
            prompts[component_name].append(prompt_tuple)
        else:
            prompts[component_name] = [prompt_tuple]
    else:
        prompts = {component_name: [prompt_tuple]}

    message.set(
        PROMPTS,
        prompts,
        add_to_output=True,
    )
