from dataclasses import dataclass, field
from typing import List, Union, Iterator, Any, Dict, Optional

from rasa.dialogue_understanding.commands import (
    Command,
    StartFlowCommand,
    SetSlotCommand,
    CancelFlowCommand,
    ChitChatAnswerCommand,
    SkipQuestionCommand,
    HumanHandoffCommand,
    KnowledgeAnswerCommand,
    ClarifyCommand,
)
from rasa.e2e_test.e2e_test_case import TestCase, TestStep
from rasa.shared.core.constants import USER


@dataclass
class ConversationStep:
    original_test_step: TestStep
    llm_commands: List[Command]
    llm_prompt: str
    failed_rephrasings: List[str] = field(default_factory=list)
    passed_rephrasings: List[str] = field(default_factory=list)
    rephrase: bool = True

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "user": self.original_test_step.text,
            "llm_commands": self._commands_to_str(),
        }
        if self.passed_rephrasings:
            data["passing_rephrasings"] = self.passed_rephrasings
        if self.failed_rephrasings:
            data["failing_rephrasings"] = self.failed_rephrasings

        return data

    def _commands_to_str(self) -> List[str]:
        output = []
        for command in self.llm_commands:
            if isinstance(command, StartFlowCommand):
                output.append(f"StartFlow({command.flow})")
            elif isinstance(command, SetSlotCommand):
                output.append(f"SetSlot({command.name}, {command.value})")
            elif isinstance(command, ClarifyCommand):
                output.append(f"Clarify({command.options})")
            elif isinstance(command, CancelFlowCommand):
                output.append("CancelFlow()")
            elif isinstance(command, ChitChatAnswerCommand):
                output.append("ChitChat()")
            elif isinstance(command, SkipQuestionCommand):
                output.append("SkipQuestion()")
            elif isinstance(command, KnowledgeAnswerCommand):
                output.append("SearchAndReply()")
            elif isinstance(command, HumanHandoffCommand):
                output.append("HumanHandoff()")
        return output

    def commands_as_string(self) -> str:
        return "\n".join(self._commands_to_str())


@dataclass
class Conversation:
    name: str
    original_e2e_test_case: TestCase
    steps: List[Union[TestStep, ConversationStep]]
    transcript: str

    def iterate_over_annotated_user_steps(
        self, rephrase: Optional[bool] = None
    ) -> Iterator[ConversationStep]:
        """Iterate over conversation steps.

        Yield each step based on the rephrase parameter.

        Args:
            rephrase: Determines whether to yield steps based on their `rephrase`
                attribute. Can be:
                - None: Yield all ConversationStep instances regardless of their
                    rephrase attribute.
                - True: Yield only those ConversationStep instances where the
                    rephrase attribute is True.
                - False: Yield only those ConversationStep instances where the
                    rephrase attribute is False.

        Yields:
            ConversationStep: The next conversation step that matches the specified
            rephrase condition.
        """
        for step in self.steps:
            if isinstance(step, ConversationStep):
                if rephrase is None:
                    yield step
                elif rephrase is not None and step.rephrase == rephrase:
                    yield step

    def get_user_messages(self) -> List[str]:
        return [
            step.original_test_step.text
            for step in self.iterate_over_annotated_user_steps()
            if step.original_test_step.text
        ]

    def get_user_messages_to_rephrase(self) -> List[str]:
        return [
            step.original_test_step.text
            for step in self.iterate_over_annotated_user_steps(rephrase=True)
            if step.original_test_step.text
        ]

    def as_dict(self) -> Dict[str, Any]:
        steps_data = []
        for step in self.steps:
            if isinstance(step, ConversationStep):
                steps_data.append(step.as_dict())
            elif isinstance(step, TestStep):
                if step.text and step.actor == USER:
                    steps_data.append({"user": step.text})
                elif step.template:
                    steps_data.append({"utter": step.template})
                elif step.text:
                    steps_data.append({"bot": step.text})

        return {
            "conversations": [
                {
                    "original_test_case": f"{self.original_e2e_test_case.file}::"
                    f"{self.original_e2e_test_case.name}",
                    "steps": steps_data,
                }
            ]
        }

    def get_number_of_rephrases(self, passing: bool) -> int:
        return sum(
            len(step.passed_rephrasings) if passing else len(step.failed_rephrasings)
            for step in self.iterate_over_annotated_user_steps()
        )

    def get_full_name(self) -> str:
        return f"{self.original_e2e_test_case.file}::{self.name}"
