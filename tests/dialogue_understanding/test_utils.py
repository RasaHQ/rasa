from typing import Dict, List, Optional, Tuple

import pytest

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.dialogue_understanding.generator import (
    MultiStepLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.nlu_command_adapter import NLUCommandAdapter
from rasa.dialogue_understanding.utils import (
    add_commands_to_message_parse_data,
    add_prompt_to_message_parse_data,
    set_record_commands_and_prompts,
)
from rasa.shared.nlu.constants import (
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
