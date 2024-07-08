from dataclasses import dataclass
from typing import List, Union, Optional

from rasa.dialogue_understanding.commands import Command
from rasa.e2e_test.e2e_test_case import TestCase, TestStep


@dataclass
class ConversationStep:
    original_test_step: TestStep
    llm_commands: List[Command]
    llm_prompt: str
    failed_rephrasings: List[str]
    passed_rephrasings: List[str]


@dataclass
class Conversation:
    name: str
    original_e2e_test_Case: TestCase
    steps: List[Union[TestStep | ConversationStep]]
    file: Optional[str] = None
