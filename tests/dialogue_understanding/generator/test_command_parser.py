from __future__ import annotations

import re
from typing import Any

import pytest

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    ChitChatAnswerCommand,
    ClarifyCommand,
    Command,
    KnowledgeAnswerCommand,
    RepeatBotMessagesCommand,
    SetSlotCommand,
    SkipQuestionCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.generator.command_parser import (
    parse_commands,
    validate_custom_commands,
)
from rasa.exceptions import ValidationError
from rasa.shared.core.flows import FlowsList
from tests.utilities import flows_from_str


class TestCommand(Command):
    @classmethod
    def command(cls) -> str:
        return "test"

    def to_dsl(self) -> str:
        return "test()"

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> TestCommand:
        return TestCommand()

    @staticmethod
    def regex_pattern() -> str:
        return r"test\(\)"

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TestCommand)


def test_additional_command_parser():
    # Given
    # When
    command = parse_commands(
        "test()",
        FlowsList(underlying_flows=[]),
        additional_commands=[TestCommand],
    )
    # Then
    assert command == [TestCommand()]


def test_update_default_command():
    # Given
    class TestCancelCommand(CancelFlowCommand):
        @staticmethod
        def regex_pattern() -> str:
            return r"Cancel\(\)"

    # Check the default command before updating
    commands = parse_commands("CancelFlow()", FlowsList(underlying_flows=[]))
    assert commands == [CancelFlowCommand()]

    # Check the default command after updating
    commands = parse_commands(
        "Cancel()",
        FlowsList(underlying_flows=[]),
        additional_commands=[TestCancelCommand],
        default_commands_to_remove=[CancelFlowCommand],
    )
    assert commands == [TestCancelCommand()]


def test_parse_commands_start_flow_command():
    # Given
    flows = flows_from_str(
        """
        flows:
          abc:
            description: some_flow
            steps:
              - id: "action1"
                action: action_listen
        """
    )

    # When
    commands = parse_commands("StartFlow('abc')", flows)

    # Then
    assert commands == [StartFlowCommand("abc")]


def test_parse_commands_set_slot_command():
    # Given
    flows = flows_from_str(
        """
        flows:
          abc:
            description: some_flow
            steps:
              - id: "action1"
                action: action_listen
        """
    )

    # When
    commands = parse_commands("SetSlot('slot_name', 'slot_value')", flows)

    # Then
    assert commands == [SetSlotCommand("slot_name", "slot_value")]


def test_parse_commands_multiple_commands():
    # Given
    flows = flows_from_str(
        """
        flows:
          abc:
            description: some_flow
            steps:
              - id: "action1"
                action: action_listen
        """
    )

    # When
    commands = parse_commands(
        "StartFlow('abc') \n SetSlot('slot_name', 'slot_value')", flows
    )

    # Then
    assert commands == [
        StartFlowCommand("abc"),
        SetSlotCommand("slot_name", "slot_value"),
    ]


def test_parse_commands_clarify_command_starts_flow():
    # Given
    flows = flows_from_str(
        """
        flows:
          abc:
            description: some_flow
            steps:
              - id: "action1"
                action: action_listen
        """
    )

    # When
    commands = parse_commands("Clarify('abc')", flows)

    # Then
    assert commands == [StartFlowCommand("abc")]


def test_parse_commands_clarify_command_optional():
    # Given
    flows = flows_from_str(
        """
        flows:
          abc:
            description: some_flow
            steps:
              - action: action_listen
          def:
            description: someother_flow
            steps:
              - action: action_listen
        """
    )
    # When
    commands = parse_commands("Clarify('abc', 'def')", flows, True)

    # Then
    assert commands == [ClarifyCommand(["abc", "def"])]


def test_parse_commands_clarify_command_optional_retuns_empty_command():
    # When
    commands = parse_commands("Clarify()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [ClarifyCommand([])]


def test_parse_commands_clarify_command_optional_retuns_empty_list():
    # When
    commands = parse_commands("Clarify()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [ClarifyCommand(options=[])]


def test_parse_commands_cancel_command():
    # When
    commands = parse_commands("CancelFlow()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [CancelFlowCommand()]


def test_parse_commands_chitchat_command():
    # When
    commands = parse_commands("ChitChat()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [ChitChatAnswerCommand()]


def test_parse_commands_skip_question_command():
    # When
    commands = parse_commands("SkipQuestion()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [SkipQuestionCommand()]


def test_parse_commands_knowledge_answer_command():
    # When
    commands = parse_commands("SearchAndReply()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [KnowledgeAnswerCommand()]


def test_parse_commands_repeat_bot_messages_command():
    # When
    commands = parse_commands("RepeatLastBotMessages()", FlowsList(underlying_flows=[]))

    # Then
    assert commands == [RepeatBotMessagesCommand()]


def test_validate_custom_commands_passed():
    # Given
    validate_custom_commands([TestCommand])


def test_validate_custom_commands_failed():
    # Given
    class InvalidCommand(Command):
        @staticmethod
        def regex_pattern() -> str:
            return r"test\(\)"

    with pytest.raises(ValidationError):
        validate_custom_commands([InvalidCommand])
