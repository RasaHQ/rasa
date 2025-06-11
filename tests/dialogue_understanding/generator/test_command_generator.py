import uuid
from typing import List, Optional, Text, Type
from unittest.mock import Mock, patch

import pytest
from pytest import CaptureFixture

from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    CannotHandleCommand,
    Command,
    ErrorCommand,
    KnowledgeAnswerCommand,
    NoopCommand,
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.dialogue_understanding.generator.command_generator import CommandGenerator
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    COMMANDS,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from tests.utilities import flows_from_str


class WackyCommandGenerator(CommandGenerator):
    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
        domain: Optional[Domain] = None,
    ) -> List[Command]:
        if message.get(TEXT) == "Hi":
            raise ValueError("Message too banal - I am quitting.")
        else:
            return [ChitChatAnswerCommand()]


async def test_command_generator_catches_processing_errors():
    generator = WackyCommandGenerator({})
    messages = [Message.build("Hi"), Message.build("What is your purpose?")]
    await generator.process(messages, FlowsList(underlying_flows=[]))
    commands = [m.get(COMMANDS) for m in messages]

    assert len(commands[0]) == 0
    assert len(commands[1]) == 1
    assert commands[1][0]["command"] == ChitChatAnswerCommand.command()


async def test_command_generator_still_throws_not_implemented_error():
    # This test can be removed if the predict_commands method stops to be abstract
    generator = CommandGenerator({})
    with pytest.raises(NotImplementedError):
        await generator.process([Message.build("test")], FlowsList([]))


@pytest.mark.parametrize(
    "commands, startable_flows, expected_commands",
    [
        ([ChitChatAnswerCommand()], FlowsList([]), [ChitChatAnswerCommand()]),
        (
            [StartFlowCommand("spam")],
            FlowsList([Flow("spam")]),
            [StartFlowCommand("spam")],
        ),
        ([StartFlowCommand("eggs")], FlowsList([Flow("spam")]), []),
        (
            [StartFlowCommand("spam"), StartFlowCommand("eggs")],
            FlowsList([Flow("spam")]),
            [StartFlowCommand("spam")],
        ),
    ],
)
def test_check_commands_against_startable_flows(
    commands: List[Command],
    startable_flows: FlowsList,
    expected_commands: List[Command],
):
    """Test that commands are correctly filtered against startable flows."""
    # Given
    generator = CommandGenerator({})
    # When
    commands = generator._check_commands_against_startable_flows(
        commands, startable_flows
    )
    # Then
    assert commands == expected_commands


@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator._check_commands_against_startable_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator.get_startable_flows"
)
async def test_command_processor_checks_flow_guards(
    mock_get_startable_flows: Mock, mock_check_commands_against_startable_flows: Mock
):
    """Test that the command processor checks flow guards."""
    # Given
    generator = WackyCommandGenerator({})
    messages = [Message.build("What is your purpose?")]
    # When
    await generator.process(messages, FlowsList([Flow("spam")]))
    # Then
    mock_get_startable_flows.assert_called_once()
    mock_check_commands_against_startable_flows.assert_called_once()


@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator._check_commands_against_startable_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator._evaluate_and_predict"
)
async def test_command_generator_no_active_flows_to_add_to_available_flows(
    mock_evaluate_and_predict: Mock,
    mock_check_commands_against_startable_flows: Mock,
):
    """Test that the command processor checks flow guards."""
    # Given
    generator = WackyCommandGenerator({})
    messages = [Message.build("What is your purpose?")]
    domain = Domain.empty()

    tracker = DialogueStateTracker.from_dict("sender_id", [])

    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
            - id: call_flow_1
              call: call_flow_1
            - id: collect_baz
              collect: baz
          call_flow_1:
            if: False
            name: call flow 1
            description: call flow 1
            steps:
            - call: call_flow_2
          call_flow_2:
            if: False
            name: call flow 2
            description: call flow 2
            steps:
            - id: collect_buzz
              collect: buzz
        """
    )

    # When
    await generator.process(messages, all_flows, tracker, domain)
    # Then
    mock_evaluate_and_predict.assert_called_once_with(
        messages[0], FlowsList([all_flows.flow_by_id("my_flow")]), tracker, domain
    )
    mock_check_commands_against_startable_flows.assert_called_once()


async def test_command_generator_add_active_flows_to_available_flows() -> None:
    """Test that the command processor checks flow guard and gets active flows."""
    # Given
    generator = WackyCommandGenerator({})

    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    user_frame_2 = UserFlowStackFrame(
        flow_id="call_flow_1",
        step_id="call_flow_1",
        frame_id="some-frame-id",
        frame_type="call",
    )
    user_frame_3 = UserFlowStackFrame(
        flow_id="call_flow_2",
        step_id="collect_buzz",
        frame_id="some-frame-id",
        frame_type="call",
    )

    stack = DialogueStack(frames=[user_frame, user_frame_2, user_frame_3])
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(stack)

    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
            - id: call_flow_1
              call: call_flow_1
            - id: collect_baz
              collect: baz
          call_flow_1:
            if: False
            name: call flow 1
            description: call flow 1
            steps:
            - call: call_flow_2
          call_flow_2:
            if: False
            name: call flow 2
            description: call flow 2
            steps:
            - id: collect_buzz
              collect: buzz
        """
    )

    # When
    result = generator.get_active_flows(all_flows, tracker)
    # Then
    expected_flows = FlowsList(
        [
            all_flows.flow_by_id("call_flow_2"),
            all_flows.flow_by_id("call_flow_1"),
            all_flows.flow_by_id("my_flow"),
        ]
    )
    assert result == expected_flows


@pytest.mark.parametrize(
    "message, expected_error_type",
    [
        (" \n\t", RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        ("", RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        ("Very long message", RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG),
    ],
)
async def test_process_invalid_messages(message: Text, expected_error_type: Text):
    # Given
    generator = WackyCommandGenerator({"user_input": {"max_characters": 10}})
    # When
    processed_messages = await generator.process(
        messages=[Message.build(text=message)], flows=FlowsList(underlying_flows=[])
    )
    # Then
    command = processed_messages[0].data["commands"][0]
    assert command["command"] == ErrorCommand.command()
    assert command["error_type"] == expected_error_type


@pytest.mark.parametrize(
    "message, expected_command_type, expected_error_type",
    [
        (" \n\t", ErrorCommand, RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        ("", ErrorCommand, RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        (
            "Very very long message",
            ErrorCommand,
            RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
        ),
        ("Chit Chat", ChitChatAnswerCommand, None),
    ],
)
async def test_evaluate_and_predict_commands(
    message: Text, expected_command_type: Type, expected_error_type: Optional[Text]
):
    # Given
    generator = WackyCommandGenerator({"user_input": {"max_characters": 20}})
    # When
    commands = await generator._evaluate_and_predict(
        message=Message.build(text=message),
        startable_flows=FlowsList(underlying_flows=[]),
    )
    # Then
    assert len(commands) == 1
    assert isinstance(commands[0], expected_command_type)
    if isinstance(commands[0], ErrorCommand):
        assert commands[0].error_type == expected_error_type


def test_command_generator_filter_commands_during_force_slot_filling() -> None:
    generator = CommandGenerator({})

    flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
              force_slot_filling: true
        """
    )

    slot_name = "foo"

    tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])
    tracker.update_stack(
        DialogueStack(
            frames=[
                UserFlowStackFrame(flow_id="my_flow", step_id="collect_foo"),
                CollectInformationPatternFlowStackFrame(collect=slot_name),
            ]
        )
    )

    commands = generator._filter_commands_during_force_slot_filling(
        [
            SetSlotCommand(name="foo", value="foo_test"),
            CancelFlowCommand(),
            ChitChatAnswerCommand(),
        ],
        flows,
        tracker,
    )

    assert len(commands) == 1
    assert commands[0] == SetSlotCommand(name="foo", value="foo_test")


def test_command_generator_filter_commands_during_force_slot_filling_cannot_handle():
    generator = CommandGenerator({})

    flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
              force_slot_filling: true
        """
    )

    slot_name = "foo"

    tracker = DialogueStateTracker.from_events(uuid.uuid4().hex, [])
    tracker.update_stack(
        DialogueStack(
            frames=[
                UserFlowStackFrame(flow_id="my_flow", step_id="collect_foo"),
                CollectInformationPatternFlowStackFrame(collect=slot_name),
            ]
        )
    )

    commands = generator._filter_commands_during_force_slot_filling(
        [CannotHandleCommand()],
        flows,
        tracker,
    )

    assert len(commands) == 1
    assert commands[0] == CannotHandleCommand()


def test_command_generator_filter_commands_during_force_slot_filling_no_tracker(
    capsys: CaptureFixture,
) -> None:
    generator = CommandGenerator({})

    flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
              force_slot_filling: true
        """
    )

    given_commands = [KnowledgeAnswerCommand()]
    commands = generator._filter_commands_during_force_slot_filling(
        given_commands,
        flows,
        None,
    )

    assert commands == given_commands
    captured = capsys.readouterr()
    assert (
        "command_generator.filter_commands_during_force_slot_filling.tracker_not_found"
        in captured.out
    )


@pytest.mark.parametrize(
    "router_commands, should_abstain",
    [
        ([SetSlotCommand(ROUTE_TO_CALM_SLOT, False)], True),
        ([NoopCommand()], True),
        ([], False),
        # not actually set by router
        ([SetSlotCommand(ROUTE_TO_CALM_SLOT, True)], False),
    ],
)
async def test_if_command_generator_should_abstain_in_coexistence(
    router_commands: List[Command],
    should_abstain: bool,
):
    """
    This test tests if the command generator should abstain from predicting
    additional commands in the coexistence setup, where router (e.g IntentBasedRouter
    or LLMBasedRouter) generated some commands.
    """
    # Given
    generator = WackyCommandGenerator({})

    # In coexistence boolean slot ROUTE_TO_CALM_SLOT needs to be defined, otherwise
    # it's routed to CALM by default
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[],
        slots=[BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[])],
    )

    # Simulate a message that already has a set of commands predicted by a router
    message = Message.build("What is your purpose?")
    message.set(
        prop=COMMANDS,
        info=[command.as_dict() for command in router_commands],
        add_to_output=True,
    )

    # When
    await generator.process(
        messages=[message],
        flows=FlowsList(underlying_flows=[]),
        tracker=tracker,
    )

    # Then
    message_command_ids = [command.get("command") for command in message.get(COMMANDS)]
    if should_abstain:
        # The commands should be the same as given router commands
        assert len(message_command_ids) == len(router_commands)
        for command in router_commands:
            assert command.command() in message_command_ids
    else:
        # The wacky command generator should predict a new command and overwrite other
        # commands
        assert len(message_command_ids) == 1
        assert ChitChatAnswerCommand().command() in message_command_ids
