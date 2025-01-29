from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator.command_parser import parse_commands
from rasa.dialogue_understanding_test.command_comparison import are_command_lists_equal
from rasa.dialogue_understanding_test.constants import (
    ACTOR_BOT,
    ACTOR_USER,
    KEY_BOT_INPUT,
    KEY_BOT_UTTERED,
    KEY_COMMANDS,
    KEY_FIXTURES,
    KEY_METADATA,
    KEY_STEPS,
    KEY_TEST_CASE,
    KEY_USER_INPUT,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_PROMPT_NAME,
    KEY_USER_PROMPT,
)


class DialogueUnderstandingOutput(BaseModel):
    """Output containing prompts and generated commands by component.

    Example of commands:
        {
            "MultiStepLLMCommandGenerator": [
                SetSlotCommand(name="slot_name", value="slot_value"),
            ],
            "NLUCommandAdapter": [
                StartFlowCommand("test_flow"),
            ]
        }

    Example of prompts:
        [
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
        ]
    """

    # Dict with component name as key and list of commands as value
    commands: Dict[str, List[Command]]
    # List of prompts
    prompts: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(frozen=True)

    def get_predicted_commands(self) -> List[Command]:
        """Get all commands from the output."""
        return [
            command
            for predicted_commands in self.commands.values()
            for command in predicted_commands
        ]

    def get_component_names_that_predicted_commands(self) -> List[str]:
        """Get all component names that have predicted commands."""
        if self.commands is None:
            return []
        return [
            component_name
            for component_name, predicted_commands in self.commands.items()
            if predicted_commands
        ]

    def get_component_name_to_user_prompts(self) -> Dict[str, List[Tuple[str, str]]]:
        """Return a dictionary of component names to a list of prompts.

        The prompts are represented as tuples of (prompt_name, user_prompt).
        """
        if self.prompts is None:
            return {}

        data: Dict[str, List[Tuple[str, str]]] = {}
        relevant_component_names = self.get_component_names_that_predicted_commands()

        for prompt_data in self.prompts:
            component_name = prompt_data[KEY_COMPONENT_NAME]

            if component_name not in relevant_component_names:
                continue

            prompt_name = prompt_data[KEY_PROMPT_NAME]
            user_prompt = prompt_data[KEY_USER_PROMPT]

            if component_name not in data:
                data[component_name] = []
            data[component_name].append((prompt_name, user_prompt))

        return data


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
                    KEY_COMMANDS: [command.to_dsl() for command in self.commands],
                }
            return {ACTOR_USER: self.text}
        elif self.actor == ACTOR_BOT:
            if self.template is not None:
                return {KEY_BOT_UTTERED: self.template}
            elif self.text is not None:
                return {KEY_BOT_INPUT: self.text}

        return {}

    @staticmethod
    def from_dict(
        step: Dict[str, Any],
        flows: FlowsList,
        custom_command_classes: List[Command] = [],
        remove_default_commands: List[str] = [],
    ) -> "DialogueUnderstandingTestStep":
        """Creates a DialogueUnderstandingTestStep from a dictionary.

        Example:
            >>> DialogueUnderstandingTestStep.from_dict({"user": "hello"})

        Args:
            step: Dictionary containing the step.
            flows: List of flows.
            custom_commands: Custom commands to use in the test case.
            remove_default_commands: Default commands to remove from the test case.

        Returns:
            DialogueUnderstandingTestStep: The constructed test step.

        Raises:
            ValueError: If the step has invalid commands that are not parseable.
        """
        # Safely extract commands from the step.
        commands = []
        for command in step.get(KEY_COMMANDS, []):
            try:
                commands.extend(
                    parse_commands(
                        command,
                        flows,
                        clarify_options_optional=True,
                        additional_commands=custom_command_classes,
                        default_commands_to_remove=remove_default_commands,
                    )
                )
            except (IndexError, ValueError) as e:
                raise ValueError(f"Failed to parse command '{command}': {e}") from e

        # Construct the DialogueUnderstandingTestStep
        return DialogueUnderstandingTestStep(
            actor=ACTOR_USER if ACTOR_USER in step else ACTOR_BOT,
            text=step.get(KEY_USER_INPUT) or step.get(KEY_BOT_INPUT),
            template=step.get(KEY_BOT_UTTERED),
            line=step.lc.line + 1 if hasattr(step, "lc") else None,
            metadata_name=step.get(KEY_METADATA, ""),
            commands=commands,
        )

    def get_predicted_commands(self) -> List[Command]:
        """Get all predicted commands from the test case."""
        if self.dialogue_understanding_output is None:
            return []

        return self.dialogue_understanding_output.get_predicted_commands()

    def has_passed(self) -> bool:
        expected_commands = self.commands or []
        predicted_commands = self.get_predicted_commands()

        return are_command_lists_equal(expected_commands, predicted_commands)

    def to_str(self) -> str:
        """Converts the test step to a readable output string."""
        if self.actor == ACTOR_BOT:
            if self.text:
                return f"{KEY_BOT_INPUT}: {self.text}"
            elif self.template:
                return f"{KEY_BOT_UTTERED}: {self.template}"

        if self.actor == ACTOR_USER:
            return f"{KEY_USER_INPUT}: {self.text}"

        return ""


class DialogueUnderstandingTestCase(BaseModel):
    name: str
    steps: List[DialogueUnderstandingTestStep] = Field(min_length=1)
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

    @staticmethod
    def from_dict(
        input_test_case: Dict[str, Any],
        flows: FlowsList,
        file: Optional[str] = None,
        custom_command_classes: List[Command] = [],
        remove_default_commands: List[str] = [],
    ) -> "DialogueUnderstandingTestCase":
        """Creates a DialogueUnderstandingTestCase from a dictionary.

        Example:
            >>> DialogueUnderstandingTestCase.from_dict({
                    "test_case": "test",
                    "steps": [{"user": "hello"}]
                })

        Args:
            input_test_case: Dictionary containing the test case.
            flows: List of flows.
            file: File name of the test case.
            custom_command_classes: Custom command classes to use in the test case.
            remove_default_commands: Default commands to remove from the test case.

        Returns:
            DialogueUnderstandingTestCase object.
        """
        steps = [
            DialogueUnderstandingTestStep.from_dict(
                step, flows, custom_command_classes, remove_default_commands
            )
            for step in input_test_case.get(KEY_STEPS, [])
        ]

        return DialogueUnderstandingTestCase(
            name=input_test_case.get(KEY_TEST_CASE, "default"),
            steps=steps,
            file=file,
            line=(
                input_test_case.lc.line + 1 if hasattr(input_test_case, "lc") else None
            ),
            fixture_names=input_test_case.get(KEY_FIXTURES),
            metadata_name=input_test_case.get(KEY_METADATA),
        )

    def to_readable_conversation(self, until_step: Optional[int] = None) -> List[str]:
        if until_step:
            steps = self.steps[:until_step]
        else:
            steps = self.steps

        return [step.to_str() for step in steps]

    def get_expected_commands(self) -> List[Command]:
        """Get all commands from the test steps."""
        return [
            command
            for step in self.iterate_over_user_steps()
            for command in (step.commands or [])
        ]

    def iterate_over_user_steps(self) -> Iterator[DialogueUnderstandingTestStep]:
        """Iterate over user steps, i.e. steps with commands."""
        for step in self.steps:
            if step.commands:
                yield step

    def get_next_user_and_bot_steps(
        self, from_index: int
    ) -> Tuple[
        Optional[DialogueUnderstandingTestStep], List[DialogueUnderstandingTestStep]
    ]:
        """Get the next user step and all following bot steps."""
        user_step = None
        bot_steps = []

        for step in self.steps[from_index:]:
            if user_step is not None and step.actor == ACTOR_USER:
                return user_step, bot_steps

            if step.actor == ACTOR_USER:
                user_step = step
            elif step.actor == ACTOR_BOT:
                bot_steps.append(step)

        return user_step, bot_steps

    def failed_user_steps(self) -> List[DialogueUnderstandingTestStep]:
        return [
            step
            for step in self.steps
            if not step.has_passed() and step.actor == ACTOR_USER
        ]


# Update forward references
DialogueUnderstandingTestStep.model_rebuild()
DialogueUnderstandingTestCase.model_rebuild()
