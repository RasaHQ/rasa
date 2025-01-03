from typing import List, Optional, Dict, Any

from pydantic import BaseModel, ConfigDict, Field

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding_test.constants import (
    ACTOR_USER,
    KEY_COMMANDS,
    ACTOR_BOT,
    KEY_TEST_CASE,
    KEY_STEPS,
    KEY_FIXTURES,
    KEY_METADATA,
    KEY_USER_INPUT,
    KEY_BOT_INPUT,
    KEY_BOT_UTTERED,
)


class DialogueUnderstandingOutput(BaseModel):
    """Output containing prompts and generated commands by component.

    Example of commands:
        {
            "MultiStepLLMCommandGenerator": [
                {"command": "set_slot", "name": "slot_name", "value": "slot_value"},
            ],
            "NLUCommandAdapter": [
                {"command": "start_flow", "name": "test_flow"},
            ]
        }

    Example of prompts:
        {
            "MultiStepLLMCommandGenerator": [
                (
                    "fill_slots_prompt",
                    {
                        "user_prompt": "<prompt content>",
                        "system_prompt": "<prompt content>"
                    }
                ),
                (
                    "handle_flows_prompt",
                    {
                        "user_prompt": "<prompt content>",
                        "system_prompt": "<prompt content>"
                    }
                ),
            ],
        }
    """

    prompts: Dict[str, tuple[str, Dict[str, str]]]
    commands: Dict[str, List[Command]]

    model_config = ConfigDict(frozen=True)

    def get_component_data(
        self, component_name: str
    ) -> tuple[Optional[tuple[str, Dict[str, str]]], List[Command]]:
        """Get both the prompts and commands for a specific component."""
        return self.prompts.get(component_name), self.commands.get(component_name, [])


class DialogueUnderstandingTestStep(BaseModel):
    actor: str
    text: Optional[str] = None
    template: Optional[str] = None
    line: Optional[int] = None
    metadata_name: Optional[str] = None
    commands: Optional[List[Command]] = None
    dialogue_understanding_output: Optional[DialogueUnderstandingOutput] = None

    def as_dict(self) -> Dict[str, Any]:
        if self.actor == ACTOR_USER:
            if self.commands:
                return {
                    KEY_USER_INPUT: self.text,
                    # TODO: The command should be converted into our DSL
                    KEY_COMMANDS: [command.as_dict() for command in self.commands],
                }
            return {ACTOR_USER: self.text}
        elif self.actor == ACTOR_BOT:
            if self.text is not None:
                return {KEY_BOT_INPUT: self.text}
            elif self.template is not None:
                return {KEY_BOT_UTTERED: self.template}

        return {}


class DialogueUnderstandingTestCase(BaseModel):
    name: str
    steps: list[DialogueUnderstandingTestStep] = Field(min_length=1)
    file: Optional[str] = None
    line: Optional[int] = None
    fixture_names: Optional[List[str]] = None
    metadata_name: Optional[str] = None

    def full_name(self) -> str:
        return f"{self.file}::{self.name}"

    def as_dict(self) -> Dict[str, Any]:
        result = {
            KEY_TEST_CASE: self.name,
            KEY_STEPS: [step.as_dict() for step in self.steps],
        }
        if self.fixture_names:
            result[KEY_FIXTURES] = self.fixture_names
        if self.metadata_name:
            result[KEY_METADATA] = self.metadata_name
        return result


# Update forward references
DialogueUnderstandingTestStep.model_rebuild()
DialogueUnderstandingTestCase.model_rebuild()
