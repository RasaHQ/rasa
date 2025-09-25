import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Type, Union

import structlog

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    Command,
    ContinueAgentCommand,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    RepeatBotMessagesCommand,
    RestartAgentCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.commands.utils import start_flow_by_name
from rasa.exceptions import ValidationError
from rasa.shared.core.flows import FlowsList

structlogger = structlog.get_logger()


DEFAULT_COMMANDS = [
    SetSlotCommand,
    StartFlowCommand,
    CancelFlowCommand,
    ChitChatAnswerCommand,
    SkipQuestionCommand,
    KnowledgeAnswerCommand,
    HumanHandoffCommand,
    ClarifyCommand,
    RepeatBotMessagesCommand,
    ContinueAgentCommand,
    RestartAgentCommand,
]


@lru_cache(maxsize=128)
def _get_compiled_pattern(pattern_str: str) -> re.Pattern:
    return re.compile(pattern_str)


def _create_default_commands(
    default_commands_to_remove: Union[List[Type[PromptCommand]], List[str]],
) -> List[Type[PromptCommand]]:
    """Return an updated list of default commands after removing the provided commands.

    Args:
        default_commands_to_remove: A list of commands to remove from the default
            commands.

    Returns:
        A copy of the default commands with the provided commands removed.
    """
    commands_after_removal = []
    for command in DEFAULT_COMMANDS:
        if (
            command not in default_commands_to_remove
            and command.__name__ not in default_commands_to_remove
        ):
            commands_after_removal.append(command)

    return commands_after_removal


def _get_additional_parsing_logic(
    command_clz: Type[PromptCommand],
) -> Optional[Callable[[PromptCommand, FlowsList], Optional[PromptCommand]]]:
    """Get additional parsing logic for a command."""
    command_to_parsing_fn_mapper: Dict[
        Type[PromptCommand], Callable[..., Optional[PromptCommand]]
    ] = {
        ClarifyCommand: _parse_clarify_command,
        SetSlotCommand: _parse_set_slot_command,
        StartFlowCommand: _parse_start_flow_command,
    }
    return command_to_parsing_fn_mapper.get(command_clz)


def validate_custom_commands(command_classes: List[Type[PromptCommand]]) -> None:
    clz_not_inheriting_from_command_clz = [
        command_clz.__name__
        for command_clz in command_classes
        if not issubclass(command_clz, Command)
    ]

    if clz_not_inheriting_from_command_clz:
        raise ValidationError(
            code="command_parser.validate_custom_commands.invalid_command",
            event_info="The additional command classes must be a subclass of the "
            "'Command' class. Please refer to the class in "
            "`rasa.dialogue_understanding.commands.command.Command`",
            invalid_commands=clz_not_inheriting_from_command_clz,
        )

    clz_not_adhering_to_prompt_command_protocol = [
        command_clz.__name__
        for command_clz in command_classes
        if not isinstance(command_clz, PromptCommand)
    ]

    if clz_not_adhering_to_prompt_command_protocol:
        raise ValidationError(
            code="command_parser.validate_custom_commands.invalid_command",
            event_info=(
                "The additional command classes must adhere to the 'PromptCommand' "
                "protocol. Please refer to the protocol in "
                "`rasa.dialogue_understanding.commands.prompt_command.PromptCommand`"
            ),
            invalid_commands=clz_not_adhering_to_prompt_command_protocol,
        )


def parse_commands(
    actions: Optional[str],
    flows: FlowsList,
    is_handle_flows_prompt: bool = False,
    additional_commands: Optional[List[Type[PromptCommand]]] = None,
    default_commands_to_remove: Optional[
        Union[List[Type[PromptCommand]], List[str]]
    ] = None,
    **kwargs: Any,
) -> List[Command]:
    """Parse a list of action commands."""
    if not actions:
        return []

    commands: List[Command] = []
    validate_custom_commands(additional_commands or [])

    default_commands = DEFAULT_COMMANDS
    if default_commands_to_remove:
        default_commands = _create_default_commands(default_commands_to_remove)

    for action in actions.strip().splitlines():
        if is_handle_flows_prompt:
            # Needed for multistep command generator.
            if (
                len(commands) >= 2
                or len(commands) == 1
                and isinstance(commands[0], ClarifyCommand)
            ):
                break
        commands.extend(
            _parse_standard_commands(default_commands, action.strip(), flows, **kwargs)
        )
        if additional_commands:
            commands.extend(
                _parse_custom_commands(
                    additional_commands, action.strip(), flows, **kwargs
                )
            )
    return commands


def _parse_standard_commands(
    standard_commands: List[Type[PromptCommand]],
    action: str,
    flows: FlowsList,
    **kwargs: Any,
) -> List[Command]:
    """Parse predefined standard commands."""
    commands: List[Command] = []
    for command_clz in standard_commands:
        pattern = _get_compiled_pattern(command_clz.regex_pattern())
        if match := pattern.search(action):
            parsed_command = command_clz.from_dsl(match, **kwargs)
            if _additional_parsing_fn := _get_additional_parsing_logic(command_clz):
                parsed_command = _additional_parsing_fn(parsed_command, flows, **kwargs)
            if parsed_command:
                commands.append(parsed_command)
    return commands


def _parse_custom_commands(
    custom_commands: List[Type[PromptCommand]],
    action: str,
    flows: FlowsList,
    **kwargs: Any,
) -> List[Command]:
    """Parse custom commands."""
    commands: List[Command] = []
    for command_clz in custom_commands:
        pattern = _get_compiled_pattern(command_clz.regex_pattern())
        if match := pattern.search(action):
            if parsed_command := command_clz.from_dsl(match, flows=flows, **kwargs):
                commands.append(parsed_command)
    return commands


def _parse_set_slot_command(
    parsed_command: Optional[SetSlotCommand], flows: FlowsList, **kwargs: Any
) -> Optional[PromptCommand]:
    """Additional parsing logic for the SetSlotCommand."""
    if not parsed_command:
        return None
    # error case where the llm tries to start a flow using a slot set
    if parsed_command.name == "flow_name":
        return start_flow_by_name(parsed_command.value, flows)
    return parsed_command


def _parse_clarify_command(
    parsed_command: Optional[ClarifyCommand], flows: FlowsList, **kwargs: Any
) -> Optional[PromptCommand]:
    """Additional parsing logic for the ClarifyCommand."""
    if not parsed_command:
        return None

    # if no options are available
    if not parsed_command.options:
        # Return the command if options are optional; else, return an empty list
        return (
            ClarifyCommand([])
            if kwargs.get("clarify_options_optional", False)
            else None
        )
    valid_options = [
        flow for flow in parsed_command.options if flow in flows.user_flow_ids
    ]
    unique_valid_options = list(set(valid_options))

    # if there is only one valid option, start the flow
    if len(unique_valid_options) == 1:
        return start_flow_by_name(valid_options[0], flows)

    # if there are multiple valid options, return a clarify command
    if len(valid_options) > 1:
        return ClarifyCommand(valid_options)
    return None


def _parse_start_flow_command(
    parsed_command: Optional[StartFlowCommand], flows: FlowsList, **kwargs: Any
) -> Optional[PromptCommand]:
    """Additional parsing logic for the StartFlowCommand."""
    if not parsed_command:
        return None
    return start_flow_by_name(parsed_command.flow, flows)
