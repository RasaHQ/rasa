from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Text

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.constants import (
    RASA_RECORD_COMMANDS_AND_PROMPTS_ENV_VAR_NAME,
)
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    COMMANDS,
    KEY_COMPONENT_NAME,
    KEY_LLM_RESPONSE_METADATA,
    KEY_PROMPT_NAME,
    KEY_SYSTEM_PROMPT,
    KEY_USER_PROMPT,
    PREDICTED_COMMANDS,
    PROMPTS,
    SET_SLOT_COMMAND,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.utils.common import get_bool_env_variable

record_commands_and_prompts = get_bool_env_variable(
    RASA_RECORD_COMMANDS_AND_PROMPTS_ENV_VAR_NAME, False
)


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
    llm_response: Optional[LLMResponse] = None,
) -> None:
    """Add prompt to the message parse data.

    Prompt is only added in case the flag 'record_commands_and_prompts' is set.
    Example of prompts in the message parse data:
        Message(data={
            PROMPTS: [
                {
                    "component_name": "MultiStepLLMCommandGenerator",
                    "prompt_name": "fill_slots_prompt",
                    "user_prompt": "...",
                    "system_prompt": "...",
                    "llm_response_metadata": { ... }
                },
                {
                    "component_name": "MultiStepLLMCommandGenerator",
                    "prompt_name": "handle_flows_prompt",
                    "user_prompt": "...",
                    "system_prompt": "...",
                    "llm_response_metadata": { ... }
                },
                {
                    "component_name": "SingleStepLLMCommandGenerator",
                    "prompt_name": "prompt_template",
                    "user_prompt": "...",
                    "system_prompt": "...",
                    "llm_response_metadata": { ... }
                }
            ]
        }
    )
    """
    # Only set prompt if the flag "record_commands_and_prompts" is set to True.
    if not record_commands_and_prompts:
        return

    # Construct the dictionary with prompt details.
    prompt_data: Dict[Text, Any] = {
        KEY_COMPONENT_NAME: component_name,
        KEY_PROMPT_NAME: prompt_name,
        KEY_USER_PROMPT: user_prompt,
        KEY_LLM_RESPONSE_METADATA: llm_response.to_dict() if llm_response else None,
        **({KEY_SYSTEM_PROMPT: system_prompt} if system_prompt else {}),
    }

    # Get or create a top-level "prompts" list.
    prompts = message.get(PROMPTS) or []
    prompts.append(prompt_data)

    # Update the message with the new prompts list.
    message.set(PROMPTS, prompts, add_to_output=True)


def _handle_via_nlu_in_coexistence(
    tracker: Optional[DialogueStateTracker], message: Message
) -> bool:
    """Check if the message should be handled by the NLU subsystem in coexistence mode."""  # noqa: E501
    if not tracker:
        return False

    if not tracker.has_coexistence_routing_slot:
        return False

    value = tracker.get_slot(ROUTE_TO_CALM_SLOT)
    if value is not None:
        return not value

    # routing slot has been reset so we need to check
    # the command issued by the Router component
    if message.get(COMMANDS):
        for command in message.get(COMMANDS):
            if (
                command.get("command") == SET_SLOT_COMMAND
                and command.get("name") == ROUTE_TO_CALM_SLOT
            ):
                return not command.get("value")

    return False
