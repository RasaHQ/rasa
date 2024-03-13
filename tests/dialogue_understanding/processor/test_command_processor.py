from typing import List, Optional
from unittest.mock import Mock, patch

import pytest
from rasa.dialogue_understanding.commands import (
    CancelFlowCommand,
    ClarifyCommand,
    Command,
    CorrectSlotsCommand,
    FreeFormAnswerCommand,
    SetSlotCommand,
    StartFlowCommand,
    ChitChatAnswerCommand,
    CannotHandleCommand,
)
from rasa.dialogue_understanding.commands.correct_slots_command import CorrectedSlot
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.correction import (
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.processor.command_processor import (
    get_commands_from_tracker,
    calculate_flow_fingerprints,
    clean_up_commands,
    contains_command,
    execute_commands,
    get_current_collect_step,
    remove_duplicated_set_slots,
    validate_state_of_commands,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    BaseFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.engine.graph import ExecutionContext
from rasa.shared.constants import RASA_PATTERN_CANNOT_HANDLE_CHITCHAT
from rasa.shared.core.constants import ACTION_TRIGGER_CHITCHAT
from rasa.shared.core.events import (
    BotUttered,
    DialogueStackUpdated,
    Event,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps import CollectInformationFlowStep
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import COMMANDS
from rasa.shared.utils.io import deep_container_fingerprint


@pytest.fixture
def collect_info_flow() -> FlowsList:
    """Return a flow that collects information."""
    return flows_from_str(
        """
        flows:
          spam:
            description: "This flow collects information."
            steps:
            - id: collect_ham
              collect: ham
              next: collect_eggs
            - id: collect_eggs
              collect: eggs
        """
    )


@pytest.fixture
def not_collect_info_flow() -> FlowsList:
    """Return a flow that does not collect information."""
    return flows_from_str(
        """
        flows:
          foo:
            steps:
            - id: first_step
              action: action_listen
        """
    )


@pytest.fixture
def user_frame_collect_eggs() -> UserFlowStackFrame:
    """Return a user frame."""
    return UserFlowStackFrame(
        flow_id="spam", step_id="collect_eggs", frame_id="some-other-frame-id"
    )


@pytest.fixture
def pattern_frame_collect_eggs() -> CollectInformationPatternFlowStackFrame:
    """Return a collect pattern frame."""
    return CollectInformationPatternFlowStackFrame(
        collect="eggs", frame_id="some-other-id"
    )


@pytest.fixture
def pattern_frame_correction() -> CorrectionPatternFlowStackFrame:
    """Return a correction pattern frame."""
    return CorrectionPatternFlowStackFrame(
        corrected_slots={"ham": 100},
    )


@pytest.mark.parametrize(
    "commands, command_type, expected_result",
    [
        ([SetSlotCommand("slot_name", "slot_value")], SetSlotCommand, True),
        ([StartFlowCommand("flow_name")], StartFlowCommand, True),
        (
            [StartFlowCommand("flow_name"), SetSlotCommand("slot_name", "slot_value")],
            StartFlowCommand,
            True,
        ),
        ([SetSlotCommand("slot_name", "slot_value")], StartFlowCommand, False),
    ],
)
def test_contains_command(commands, command_type, expected_result):
    """Test if commands contains a command of a given type."""
    # When
    result = contains_command(commands, command_type)
    # Then
    assert result == expected_result


def test_get_commands_from_tracker(tracker: DialogueStateTracker):
    """Test if commands are correctly extracted from tracker."""
    # When
    commands = get_commands_from_tracker(tracker)
    # Then
    assert isinstance(commands[0], StartFlowCommand)
    assert commands[0].command() == "start flow"
    assert commands[0].flow == "foo"


def test_get_commands_from_tracker_with_no_messages():
    """Test no commands are extracted from tracker when there is no latest message."""
    # Given
    tracker = DialogueStateTracker.from_events("test", evts=[])
    # When
    commands = get_commands_from_tracker(tracker)
    # Then
    assert commands == []


@pytest.mark.parametrize(
    "commands",
    [
        [CancelFlowCommand()],
        [StartFlowCommand("flow_name")],
        [SetSlotCommand("slot_name", "slot_value")],
        [StartFlowCommand("flow_name"), SetSlotCommand("slot_name", "slot_value")],
        [FreeFormAnswerCommand(), SetSlotCommand("slot_name", "slot_value")],
        [
            FreeFormAnswerCommand(),
            FreeFormAnswerCommand(),
            StartFlowCommand("flow_name"),
        ],
        [CorrectSlotsCommand([])],
        [CorrectSlotsCommand([]), StartFlowCommand("flow_name")],
    ],
)
def test_validate_state_of_commands(commands):
    """Test if commands are correctly validated."""
    # Then
    try:
        validate_state_of_commands(commands)
    except ValueError:
        pytest.fail("validate_state_of_commands raised ValueError unexpectedly")


@pytest.mark.parametrize(
    "commands",
    [
        [CancelFlowCommand(), CancelFlowCommand()],
        [StartFlowCommand("flow_name"), FreeFormAnswerCommand()],
        [CorrectSlotsCommand([]), CorrectSlotsCommand([])],
    ],
)
def test_validate_state_of_commands_raises_exception(commands):
    """Test if commands are correctly validated."""
    # Then
    with pytest.raises(ValueError):
        validate_state_of_commands(commands)


def test_calculate_flow_fingerprints(all_flows: FlowsList):
    """Test if flow fingerprints are correctly calculated."""
    # Given
    expected_fingerprints = {
        f.id: deep_container_fingerprint(f.as_json()) for f in all_flows
    }
    # When
    fingerprints = calculate_flow_fingerprints(all_flows)
    # Then
    assert fingerprints == expected_fingerprints


def test_execute_commands(all_flows: FlowsList):
    """Test if commands are correctly executed."""
    # Given
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered(
                "start foo", None, None, {COMMANDS: [StartFlowCommand("foo").as_dict()]}
            )
        ],
    )
    # When
    events = execute_commands(tracker, all_flows, Mock())
    # Then
    assert len(events) == 2

    assert isinstance(events[0], SlotSet)
    assert events[0].key == "flow_hashes"
    assert events[0].value.keys() == {"foo", "bar"}

    assert isinstance(events[1], DialogueStackUpdated)
    updated_stack = tracker.stack.update_from_patch(events[1].update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo"
    assert frame.step_id == "START"
    assert frame.frame_type == "regular"


@pytest.mark.parametrize(
    "events, expected_events",
    [
        ([SlotSet(key="spam")], [SlotSet(key="spam")]),
        (
            [SlotSet(key="spam"), SlotSet(key="eggs")],
            [SlotSet(key="spam"), SlotSet(key="eggs")],
        ),
        ([SlotSet(key="spam"), SlotSet(key="spam")], [SlotSet(key="spam")]),
        (
            [SlotSet(key="spam"), UserUttered("ham"), BotUttered("eggs")],
            [SlotSet(key="spam"), UserUttered("ham"), BotUttered("eggs")],
        ),
    ],
)
def test_remove_duplicated_set_slots(events: List[Event], expected_events: List[Event]):
    """Test if duplicated set slots are correctly removed."""
    # When
    deduplicated_events = remove_duplicated_set_slots(events)
    # Then
    assert deduplicated_events == expected_events


def test_get_current_collect_step(
    collect_info_flow: FlowsList,
    user_frame_collect_eggs: UserFlowStackFrame,
    pattern_frame_collect_eggs: CollectInformationPatternFlowStackFrame,
):
    # Given
    stack = DialogueStack(frames=[user_frame_collect_eggs, pattern_frame_collect_eggs])
    # When
    current_collect_step = get_current_collect_step(stack, collect_info_flow)
    # Then
    assert isinstance(current_collect_step, CollectInformationFlowStep)
    assert current_collect_step.collect == "eggs"


@pytest.mark.parametrize(
    "frames, flows",
    [
        ([], collect_info_flow),
        ([user_frame_collect_eggs], collect_info_flow),
        ([pattern_frame_collect_eggs], collect_info_flow),
        ([user_frame_collect_eggs, pattern_frame_collect_eggs], not_collect_info_flow),
    ],
)
def test_get_current_collect_step_returns_none(
    frames: List[Optional[BaseFlowStackFrame]],
    flows: FlowsList,
):
    # Given
    stack = DialogueStack(frames=frames)
    # When
    current_collect_step = get_current_collect_step(stack, flows)
    # Then
    assert current_collect_step is None


@pytest.mark.parametrize(
    "commands, expected_clean_commands",
    [
        ([SetSlotCommand("slot_name", "foo")], [SetSlotCommand("slot_name", "foo")]),
        (
            [SetSlotCommand("slot_a", "foo"), SetSlotCommand("slot_b", "bar")],
            [SetSlotCommand("slot_a", "foo"), SetSlotCommand("slot_b", "bar")],
        ),
        ([SetSlotCommand("eggs", "scrambled")], [SetSlotCommand("eggs", "scrambled")]),
        (
            [SetSlotCommand("ham", "prosciutto")],
            [
                CorrectSlotsCommand(
                    corrected_slots=[CorrectedSlot(name="ham", value="prosciutto")]
                )
            ],
        ),
        (
            [SetSlotCommand("ham", "prosciutto"), SetSlotCommand("ham", "serrano")],
            [
                CorrectSlotsCommand(
                    corrected_slots=[
                        CorrectedSlot(name="ham", value="prosciutto"),
                        CorrectedSlot(name="ham", value="serrano"),
                    ]
                )
            ],
        ),
        (
            [SetSlotCommand("ham", "prosciutto"), SetSlotCommand("ham", "prosciutto")],
            [
                CorrectSlotsCommand(
                    corrected_slots=[
                        CorrectedSlot(name="ham", value="prosciutto"),
                        CorrectedSlot(name="ham", value="prosciutto"),
                    ]
                )
            ],
        ),
        ([FreeFormAnswerCommand()], [FreeFormAnswerCommand()]),
        (
            [StartFlowCommand("flow_name"), FreeFormAnswerCommand()],
            [FreeFormAnswerCommand(), StartFlowCommand("flow_name")],
        ),
        ([CancelFlowCommand()], [CancelFlowCommand()]),
        ([CancelFlowCommand(), CancelFlowCommand()], [CancelFlowCommand()]),
        # keep only first if there are multiple clarify commands
        (
            [ClarifyCommand(["a", "b", "c"]), ClarifyCommand(["d", "e"])],
            [ClarifyCommand(["a", "b", "c"])],
        ),
        # drop clarify command if there are other commands
        (
            [StartFlowCommand("flow_name"), ClarifyCommand(["x", "y"])],
            [StartFlowCommand("flow_name")],
        ),
        # drop clarify command of flow that is already on the tracker
    ],
)
def test_clean_up_commands(
    collect_info_flow: FlowsList,
    user_frame_collect_eggs: UserFlowStackFrame,
    pattern_frame_collect_eggs: CollectInformationPatternFlowStackFrame,
    commands: List[Command],
    expected_clean_commands: List[Command],
):

    stack = DialogueStack(frames=[user_frame_collect_eggs, pattern_frame_collect_eggs])

    tracker_eggs = DialogueStateTracker.from_events(sender_id="test", evts=[])
    tracker_eggs.update_stack(stack)
    # When
    with patch(
        (
            "rasa.dialogue_understanding.processor."
            "command_processor.filled_slots_for_active_flow"
        ),
        Mock(return_value=({"ham"}, "egg")),
    ):
        clean_commands = clean_up_commands(
            commands, tracker_eggs, collect_info_flow, Mock()
        )

    # Then
    assert clean_commands == expected_clean_commands


@pytest.mark.parametrize(
    "commands, expected_clean_commands",
    [
        ([SetSlotCommand("ham", "100")], []),
        ([SetSlotCommand("egg", "some_value")], []),
        ([SetSlotCommand("eggs", "scrambled")], [SetSlotCommand("eggs", "scrambled")]),
        (
            [
                SetSlotCommand("ham", "100"),
                SetSlotCommand("ham", 100),
                SetSlotCommand("eggs", "scrambled"),
            ],
            [SetSlotCommand("eggs", "scrambled")],
        ),
    ],
)
def test_clean_up_commands_with_correction_pattern_on_stack(
    collect_info_flow: FlowsList,
    pattern_frame_correction: CorrectionPatternFlowStackFrame,
    user_frame_collect_eggs: CollectInformationPatternFlowStackFrame,
    commands: List[Command],
    expected_clean_commands: List[Command],
):
    stack = DialogueStack(frames=[user_frame_collect_eggs, pattern_frame_correction])

    tracker_eggs = DialogueStateTracker.from_events(
        sender_id="test",
        evts=[],
        slots=[TextSlot("egg", mappings=[], initial_value="some_value")],
    )
    tracker_eggs.update_stack(stack)
    # When
    with patch(
        (
            "rasa.dialogue_understanding.processor."
            "command_processor.filled_slots_for_active_flow"
        ),
        Mock(return_value=({"ham", "egg"}, "spam")),
    ):
        clean_commands = clean_up_commands(commands, tracker_eggs, collect_info_flow)

    # Then
    assert clean_commands == expected_clean_commands


@pytest.mark.parametrize(
    "commands, expected_clean_commands",
    [
        ([StartFlowCommand("spam")], []),
        ([StartFlowCommand("eggs")], [StartFlowCommand("eggs")]),
    ],
)
def test_clean_up_commands_with_start_flow(
    collect_info_flow: FlowsList,
    user_frame_collect_eggs: CollectInformationPatternFlowStackFrame,
    commands: List[Command],
    expected_clean_commands: List[Command],
):
    stack = DialogueStack(frames=[user_frame_collect_eggs])

    tracker_eggs = DialogueStateTracker.from_events(sender_id="test", evts=[])
    tracker_eggs.update_stack(stack)
    # When
    with patch(
        (
            "rasa.dialogue_understanding.processor."
            "command_processor.filled_slots_for_active_flow"
        ),
        Mock(return_value=({"ham"}, "spam")),
    ):
        clean_commands = clean_up_commands(
            commands, tracker_eggs, collect_info_flow, Mock()
        )

    # Then
    assert clean_commands == expected_clean_commands


@pytest.mark.parametrize(
    (
        "commands,"
        "expected_clean_commands,"
        "defines_intentless_policy,"
        "uses_action_trigger_chitchat"
    ),
    [
        ([ChitChatAnswerCommand()], [ChitChatAnswerCommand()], True, True),
        ([ChitChatAnswerCommand()], [ChitChatAnswerCommand()], False, False),
        (
            [ChitChatAnswerCommand()],
            [CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_CHITCHAT)],
            False,
            True,
        ),
        ([ChitChatAnswerCommand()], [ChitChatAnswerCommand()], True, False),
    ],
)
@patch("rasa.shared.core.flows.flows_list.FlowsList.flow_by_id")
@patch("rasa.engine.graph.ExecutionContext.has_node")
def test_clean_up_chitchat_commands(
    mock_execution_context_has_node: Mock,
    mock_flow_by_id: Mock,
    commands,
    expected_clean_commands,
    defines_intentless_policy,
    uses_action_trigger_chitchat,
):
    # Given
    # mock getting the pattern_chitchat flow
    flows = FlowsList(underlying_flows=[])
    mock_pattern_chitchat = Mock()
    mock_pattern_chitchat.has_action_step = Mock(
        return_value=uses_action_trigger_chitchat
    )
    mock_flow_by_id.return_value = mock_pattern_chitchat
    # mock the presence of the intentless policy in the execution context
    execution_context = ExecutionContext(Mock())
    mock_execution_context_has_node.return_value = defines_intentless_policy
    # mock the tracker
    tracker = DialogueStateTracker.from_events(sender_id="test", evts=[])

    # When
    clean_commands = clean_up_commands(commands, tracker, flows, execution_context)

    # Then
    mock_execution_context_has_node.assert_called_once()
    mock_pattern_chitchat.has_action_step.assert_called_once_with(
        ACTION_TRIGGER_CHITCHAT
    )
    assert clean_commands == expected_clean_commands
