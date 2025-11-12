import uuid
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
import structlog.testing

from rasa.dialogue_understanding.commands import (
    Command,
    NoopCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.generator import (
    MultiStepLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.dialogue_understanding.utils import (
    DEFAULT_MAX_CLARIFICATION_OPTIONS,
    MAX_CLARIFICATION_OPTIONS_SLOT_NAME,
    _handle_via_nlu_in_coexistence,
    add_commands_to_message_parse_data,
    add_prompt_to_message_parse_data,
    assemble_options_string,
    limit_clarification_options,
    set_record_commands_and_prompts,
)
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.slots import BooleanSlot
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
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.llm.llm_response import LLMResponse
from tests.utilities import filter_logs


@pytest.fixture
def mock_tracker() -> MagicMock:
    """Create a mock tracker for testing."""
    tracker = MagicMock(spec=DialogueStateTracker)
    tracker.slots = {}
    return tracker


@pytest.mark.parametrize(
    "current_commands, component_name, expected_commands",
    [
        (
            {NLUCommandAdapter.__name__: [{"command": "cancel flow"}]},
            NLUCommandAdapter.__name__,
            {
                NLUCommandAdapter.__name__: [
                    {"command": "cancel flow"},
                    StartFlowCommand("test").as_dict(),
                ]
            },
        ),
        (
            None,
            NLUCommandAdapter.__name__,
            {NLUCommandAdapter.__name__: [StartFlowCommand("test").as_dict()]},
        ),
        (
            {SingleStepLLMCommandGenerator.__name__: [{"command": "cancel flow"}]},
            NLUCommandAdapter.__name__,
            {
                NLUCommandAdapter.__name__: [StartFlowCommand("test").as_dict()],
                SingleStepLLMCommandGenerator.__name__: [{"command": "cancel flow"}],
            },
        ),
    ],
)
def test_add_commands_to_message_parse_data(
    current_commands: Optional[Dict[str, List[Dict[str, str]]]],
    component_name: str,
    expected_commands: Dict[str, List[Dict[str, str]]],
):
    # Given
    message = Message(data={TEXT: "some message", PREDICTED_COMMANDS: current_commands})
    commands = [StartFlowCommand("test")]

    # When
    with set_record_commands_and_prompts():
        add_commands_to_message_parse_data(message, component_name, commands)

    # Then
    assert message.get(PREDICTED_COMMANDS) == expected_commands


@pytest.mark.parametrize(
    "current_prompts, component_name, system_prompt, expected_prompts",
    [
        (
            [
                {
                    KEY_COMPONENT_NAME: MultiStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt_template",
                    KEY_USER_PROMPT: "prompt content",
                    KEY_SYSTEM_PROMPT: "system prompt",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
            ],
            MultiStepLLMCommandGenerator.__name__,
            None,
            [
                {
                    KEY_COMPONENT_NAME: MultiStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt_template",
                    KEY_USER_PROMPT: "prompt content",
                    KEY_SYSTEM_PROMPT: "system prompt",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
                {
                    KEY_COMPONENT_NAME: MultiStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt name",
                    KEY_USER_PROMPT: "test prompt",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
            ],
        ),
        (
            None,
            SingleStepLLMCommandGenerator.__name__,
            "system prompt content",
            [
                {
                    KEY_COMPONENT_NAME: SingleStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt name",
                    KEY_USER_PROMPT: "test prompt",
                    KEY_SYSTEM_PROMPT: "system prompt content",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
            ],
        ),
        (
            [
                {
                    KEY_COMPONENT_NAME: SingleStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt_template",
                    KEY_USER_PROMPT: "prompt content",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
            ],
            MultiStepLLMCommandGenerator.__name__,
            None,
            [
                {
                    KEY_COMPONENT_NAME: SingleStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt_template",
                    KEY_USER_PROMPT: "prompt content",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
                {
                    KEY_COMPONENT_NAME: MultiStepLLMCommandGenerator.__name__,
                    KEY_PROMPT_NAME: "prompt name",
                    KEY_USER_PROMPT: "test prompt",
                    KEY_LLM_RESPONSE_METADATA: LLMResponse(
                        id="mock-id", choices=["some message"], created=123456
                    ).to_dict(),
                },
            ],
        ),
    ],
)
def test_add_prompt_to_message_parse_data(
    current_prompts: Optional[Dict[str, List[Tuple[str, str]]]],
    component_name: str,
    system_prompt: Optional[str],
    expected_prompts: Dict[str, List[Tuple[str, str]]],
):
    # Given
    message = Message(data={TEXT: "some message", PROMPTS: current_prompts})
    user_prompt = "test prompt"
    prompt_name = "prompt name"
    llm_response = LLMResponse(id="mock-id", choices=["some message"], created=123456)

    # When
    with set_record_commands_and_prompts():
        add_prompt_to_message_parse_data(
            message,
            component_name,
            prompt_name,
            user_prompt,
            system_prompt,
            llm_response,
        )

    # Then
    assert message.get(PROMPTS) == expected_prompts


@pytest.mark.parametrize(
    "tracker_defined,"
    "tracker_has_coexistence_slot,"
    "tracker_route_to_calm_slot_value,"
    "message_commands,"
    "expected_output",
    [
        # No tracker at all
        (False, None, None, None, False),
        # Tracker without coexistence slot
        (True, False, None, None, False),
        # Tracker has coexistence slot and slot is True
        # -> route to CALM
        # -> return False
        (True, True, True, None, False),
        # Tracker has coexistence slot and slot is False
        # -> route to NLU
        # -> return True
        (True, True, False, None, True),
        # Tracker slot is None, but SetSlotCommand in message sets slot to True
        # -> route to CALM
        # -> return False
        (True, True, None, [SetSlotCommand(ROUTE_TO_CALM_SLOT, True)], False),
        # Tracker slot is None, but SetSlotCommand in message sets slot to False
        # -> route to NLU
        # -> return True
        (True, True, None, [SetSlotCommand(ROUTE_TO_CALM_SLOT, False)], True),
        # Tracker slot is None, NoopCommand found in message
        # -> route to NLU
        # -> return True
        (True, True, None, [NoopCommand()], True),
        # Tracker slot is None and no usable commands
        # -> default to CALM
        # -> return False
        (True, True, None, [], False),
    ],
)
def test_handle_via_nlu_in_coexistence(
    tracker_defined: bool,
    tracker_has_coexistence_slot: Optional[bool],
    tracker_route_to_calm_slot_value: Optional[bool],
    message_commands: List[Command],
    expected_output: bool,
):
    # Given

    # Setup tracker
    if not tracker_defined:
        tracker = None
    else:
        if tracker_has_coexistence_slot:
            slots = [
                BooleanSlot(
                    ROUTE_TO_CALM_SLOT,
                    mappings=[],
                    initial_value=tracker_route_to_calm_slot_value,
                )
            ]
        else:
            slots = []

        tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [], slots=slots)

    # Simulate a message that already has a set of commands predicted by a router
    message = Message.build("What is your purpose?")
    if message_commands:
        message.set(
            prop=COMMANDS,
            info=[command.as_dict() for command in message_commands],
            add_to_output=True,
        )

    # When
    result = _handle_via_nlu_in_coexistence(tracker, message)

    # Then
    assert result is expected_output


@pytest.mark.parametrize(
    "names, conjunction, expected_output",
    [
        # Single item
        (["apple"], "and", "apple"),
        (["banana"], "or", "banana"),
        # Two items
        (["apple", "banana"], "and", "apple and banana"),
        (["red", "blue"], "or", "red or blue"),
        (["cat", "dog"], "but", "cat but dog"),
        # Three items
        (["apple", "banana", "cherry"], "and", "apple, banana and cherry"),
        (["red", "blue", "green"], "or", "red, blue or green"),
        (["cat", "dog", "bird"], "but", "cat, dog but bird"),
        # Four items
        (
            ["winter", "spring", "summer", "fall"],
            "and",
            "winter, spring, summer and fall",
        ),
        (["north", "south", "east", "west"], "or", "north, south, east or west"),
        # Empty list
        ([], "and", ""),
        # Custom conjunction
        (["option1", "option2", "option3"], "plus", "option1, option2 plus option3"),
        # Using default conjunction
        (["apple"], None, "apple"),
        (["apple", "banana"], None, "apple and banana"),
        (["apple", "banana", "cherry"], None, "apple, banana and cherry"),
        ([], None, ""),
    ],
)
def test_assemble_options_string(
    names: List[str], conjunction: Optional[str], expected_output: str
):
    """Test assemble_options_string function with various inputs."""
    if conjunction is None:
        result = assemble_options_string(names)
    else:
        result = assemble_options_string(names, conjunction)

    assert result == expected_output


@pytest.mark.parametrize(
    "slot_value,names,expected_result",
    [
        # Limit is less than available names
        (
            3,
            ["flow1", "flow2", "flow3", "flow4", "flow5"],
            ["flow1", "flow2", "flow3"],
        ),
        # Limit exceeds available names
        (
            10,
            ["flow1", "flow2", "flow3"],
            ["flow1", "flow2", "flow3"],
        ),
        # Empty names list
        (
            2,
            [],
            [],
        ),
    ],
)
def test_limit_clarification_options_with_slot(
    mock_tracker: MagicMock,
    slot_value: int,
    names: List[str],
    expected_result: List[str],
) -> None:
    """Test limiting clarification options with valid slot values."""
    mock_tracker.get_slot.return_value = slot_value

    result = limit_clarification_options(mock_tracker, names)

    assert result == expected_result


@pytest.mark.parametrize(
    "names,expected_result",
    [
        (
            ["flow1", "flow2", "flow3", "flow4", "flow5", "flow6"],
            ["flow1", "flow2", "flow3"],
        ),
        (
            ["flow1"],
            ["flow1"],
        ),
        (
            [],
            [],
        ),
    ],
)
def test_limit_clarification_options_without_slot(
    mock_tracker: MagicMock,
    names: List[str],
    expected_result: List[str],
) -> None:
    """Test that no initial slot value falls back to default."""
    mock_tracker.get_slot.return_value = None
    result = limit_clarification_options(mock_tracker, names)

    assert result == expected_result
    assert len(result) <= DEFAULT_MAX_CLARIFICATION_OPTIONS


@pytest.mark.parametrize(
    "invalid_value,names",
    [
        (
            "not_a_number",
            ["flow1", "flow2", "flow3", "flow4", "flow5"],
        ),
        (None, ["flow1", "flow2", "flow3"]),
        ([], ["flow1", "flow2"]),
        ({}, ["flow1"]),
    ],
)
def test_limit_clarification_options_with_invalid_slot_value(
    mock_tracker: MagicMock,
    invalid_value,
    names: List[str],
) -> None:
    """Test that invalid slot values fall back to default slot value."""
    mock_tracker.get_slot.return_value = invalid_value

    with structlog.testing.capture_logs() as caplog:
        result = limit_clarification_options(mock_tracker, names)
        logs = filter_logs(
            caplog,
            "utils.limit_clarification_options.invalid_slot_value",
            "debug",
            [
                f"Slot '{MAX_CLARIFICATION_OPTIONS_SLOT_NAME}' has invalid value. "
                f"Falling back to default '{DEFAULT_MAX_CLARIFICATION_OPTIONS}'."
            ],
        )
        assert len(logs) == 1

    assert result == names[:DEFAULT_MAX_CLARIFICATION_OPTIONS]
    assert len(result) <= DEFAULT_MAX_CLARIFICATION_OPTIONS
