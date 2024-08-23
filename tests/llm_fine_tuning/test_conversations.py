import unittest
from unittest.mock import MagicMock

from rasa.dialogue_understanding.commands import (
    StartFlowCommand,
    SetSlotCommand,
    ClarifyCommand,
)
from rasa.e2e_test.e2e_test_case import TestCase, TestStep
from rasa.llm_fine_tuning.conversations import ConversationStep, Conversation


class TestConversationStep(unittest.TestCase):
    def setUp(self):
        self.test_step = MagicMock(spec=TestStep)
        self.test_step.text = "user input"
        self.commands = [
            StartFlowCommand(flow="test_flow"),
            SetSlotCommand(name="slot_name", value="slot_value"),
            ClarifyCommand(options=["option1", "option2"]),
        ]

    def test_as_dict(self):
        conversation_step = ConversationStep(
            self.test_step, self.commands, "llm_prompt"
        )
        expected = {
            "user": "user input",
            "llm_commands": [
                "StartFlow(test_flow)",
                "SetSlot(slot_name, slot_value)",
                "Clarify(['option1', 'option2'])",
            ],
        }

        assert conversation_step.as_dict() == expected

    def test_as_dict_with_rephrasings(self):
        conversation_step = ConversationStep(
            self.test_step,
            self.commands,
            "llm_prompt",
            failed_rephrasings=["fail1"],
            passed_rephrasings=["pass1"],
        )

        expected = {
            "user": "user input",
            "llm_commands": [
                "StartFlow(test_flow)",
                "SetSlot(slot_name, slot_value)",
                "Clarify(['option1', 'option2'])",
            ],
            "passing_rephrasings": ["pass1"],
            "failing_rephrasings": ["fail1"],
        }

        assert conversation_step.as_dict() == expected


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.test_case = MagicMock(spec=TestCase)
        self.test_case.file = "test_file"
        self.test_case.name = "test_case"
        self.test_step = MagicMock(spec=TestStep)
        self.test_step.text = "some text"
        self.test_step.actor = "bot"
        self.test_step.template = "some template"
        self.conv_step = ConversationStep(
            self.test_step, [], "llm_prompt", ["failed_1"], ["passing_1", "passing_2"]
        )

        self.conversation = Conversation(
            name="test_conversation",
            original_e2e_test_case=self.test_case,
            steps=[self.conv_step, self.test_step, self.conv_step],
            transcript="transcript text",
        )

    def test_iterate_over_user_steps(self):
        steps = list(self.conversation.iterate_over_annotated_user_steps())

        assert len(steps) == 2
        assert steps[0] == self.conv_step

    def test_get_user_messages(self):
        messages = self.conversation.get_user_messages()

        assert messages == ["some text", "some text"]

    def test_as_dict(self):
        expected = {
            "conversations": [
                {
                    "original_test_case": "test_file::test_case",
                    "steps": [
                        {
                            "llm_commands": [],
                            "user": "some text",
                            "failing_rephrasings": ["failed_1"],
                            "passing_rephrasings": ["passing_1", "passing_2"],
                        },
                        {"utter": "some template"},
                        {
                            "llm_commands": [],
                            "user": "some text",
                            "failing_rephrasings": ["failed_1"],
                            "passing_rephrasings": ["passing_1", "passing_2"],
                        },
                    ],
                }
            ]
        }

        result = self.conversation.as_dict()

        assert result == expected

    def test_get_number_of_rephrases(self):
        failed_rephrases = self.conversation.get_number_of_rephrases(False)
        assert failed_rephrases == 2

        passed_rephrases = self.conversation.get_number_of_rephrases(True)
        assert passed_rephrases == 4
