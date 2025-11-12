import typing
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Text

import structlog

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
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.utils.common import get_bool_env_variable

if typing.TYPE_CHECKING:
    from rasa.dialogue_understanding.commands import Command

record_commands_and_prompts = get_bool_env_variable(
    RASA_RECORD_COMMANDS_AND_PROMPTS_ENV_VAR_NAME, False
)

structlogger = structlog.get_logger()

DEFAULT_MAX_CLARIFICATION_OPTIONS = 3
MAX_CLARIFICATION_OPTIONS_SLOT_NAME = "max_clarification_options"


@contextmanager
def set_record_commands_and_prompts() -> Generator:
    global record_commands_and_prompts
    record_commands_and_prompts = True
    try:
        yield
    finally:
        record_commands_and_prompts = False


def add_commands_to_message_parse_data(
    message: Message, component_name: str, commands: List["Command"]
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
    from rasa.dialogue_understanding.commands import (
        NoopCommand,
        SetSlotCommand,
    )

    if not tracker:
        return False

    commands = message.get(COMMANDS, [])

    # If coexistence routing slot is not active, this setup doesn't
    # support dual routing -> default to CALM
    if not tracker.has_coexistence_routing_slot:
        structlogger.debug(
            "utils.handle_via_nlu_in_coexistence"
            ".tracker_missing_route_session_to_calm_slot",
            event_info=(
                f"Tracker doesn't have the '{ROUTE_TO_CALM_SLOT}' slot.Routing to CALM."
            ),
            route_session_to_calm=commands,
        )
        return False

    # Check if the routing decision is stored in the tracker slot
    # If slot is true -> route to CALM
    # If slot is false -> route to DM1
    value = tracker.get_slot(ROUTE_TO_CALM_SLOT)
    if value is not None:
        structlogger.debug(
            "utils.handle_via_nlu_in_coexistence"
            ".tracker_route_session_to_calm_slot_value",
            event_info=(
                f"Tracker slot '{ROUTE_TO_CALM_SLOT}' set to '{value}'. "
                f"Routing to "
                f"{'CALM' if value else 'NLU system'}."
            ),
            route_session_to_calm_value_in_tracker=value,
        )
        return not value

    # Non-sticky routing to DM1 is only allowed if NoopCommand is the sole predicted
    # command. In that case, route to DM1
    if len(commands) == 1 and commands[0].get("command") == NoopCommand.command():
        structlogger.debug(
            "utils.handle_via_nlu_in_coexistence.noop_command_detected",
            event_info="NoopCommand found. Routing to NLU system non-sticky.",
            commands=commands,
        )
        return True

    # If the slot was reset (e.g. new session), try to infer routing from
    # attached commands. Look for a SetSlotCommand targeting the ROUTE_TO_CALM_SLOT
    for command in message.get(COMMANDS, []):
        # If slot is true -> route to CALM
        # If slot is false -> route to DM1
        if (
            command.get("command") == SetSlotCommand.command()
            and command.get("name") == ROUTE_TO_CALM_SLOT
        ):
            structlogger.debug(
                "utils.handle_via_nlu_in_coexistence.set_slot_command_detected",
                event_info=(
                    f"SetSlotCommand setting the '{ROUTE_TO_CALM_SLOT}' to "
                    f"'{command.get('value')}'. "
                    f"Routing to "
                    f"{'CALM' if command.get('value') else 'NLU system'}."
                ),
                commands=commands,
            )
            return not command.get("value")

    # If no routing info is available -> default to CALM
    structlogger.debug(
        "utils.handle_via_nlu_in_coexistence.no_routing_info_available",
        event_info="No routing info available. Routing to CALM.",
        commands=commands,
    )
    return False


def assemble_options_string(names: List[str], conjunction: str = "and") -> str:
    """Concatenate options to a human-readable string."""
    option_message = ""
    for i, name in enumerate(names):
        if i == 0:
            option_message += name
        elif i == len(names) - 1:
            option_message += f" {conjunction} {name}"
        else:
            option_message += f", {name}"
    return option_message


def limit_clarification_options(
    tracker: "DialogueStateTracker", names: List[str]
) -> List[str]:
    """Limit the number of clarification options based on the slot value.

    Args:
        tracker: The dialogue state tracker.
        names: The list of clarification option names.

    Returns:
        The limited list of names, or the original list if no valid limit is set.
    """
    max_options_slot = tracker.get_slot(MAX_CLARIFICATION_OPTIONS_SLOT_NAME)
    try:
        max_options = int(max_options_slot)
    except (ValueError, TypeError):
        structlogger.debug(
            "utils.limit_clarification_options.invalid_slot_value",
            slot_value=max_options_slot,
            event_info=(
                f"Slot '{MAX_CLARIFICATION_OPTIONS_SLOT_NAME}' has invalid value. "
                f"Falling back to default '{DEFAULT_MAX_CLARIFICATION_OPTIONS}'."
            ),
        )
        max_options = DEFAULT_MAX_CLARIFICATION_OPTIONS

    return names[:max_options]
