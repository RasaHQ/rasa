from typing import Any, Dict, Iterator, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import MonkeyPatch

from rasa.agents.constants import A2A_AGENT_CONTEXT_ID_KEY
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.schemas import AgentOutput
from rasa.agents.schemas.agent_input import AgentInput, AgentInputSlot
from rasa.core.available_agents import AvailableAgents
from rasa.core.policies.flows.agent_executor import (
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    MAX_AGENT_RETRIES,
    SLOTS_EXCLUDED_FOR_AGENT,
    _call_agent_with_retry,
    _create_action_prediction,
    _create_agent_request_user_input_prediction,
    _create_send_text_prediction,
    _handle_agent_completed,
    _handle_agent_fatal_error,
    _handle_agent_input_required,
    _handle_agent_unknown_status,
    _handle_resume_interrupted_agent,
    _prepare_agent_input,
    _prepare_slots_for_agent,
    _reset_slots_covered_by_exit_if,
    _update_agent_events,
    _update_agent_input_metadata_with_events,
    remove_agent_stack_frame,
    run_agent,
)
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    PauseFlowReturnPrediction,
)
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    UserFlowStackFrame,
)
from rasa.shared.core.constants import (
    ACTION_AGENT_REQUEST_USER_INPUT_NAME,
    ACTION_METADATA_MESSAGE_KEY,
    ACTION_METADATA_TEXT_KEY,
    ACTION_SEND_TEXT_NAME,
    DEFAULT_SLOT_NAMES,
    FLOW_HASHES_SLOT,
    KNOWLEDGE_BASE_SLOT_NAMES,
    REQUESTED_SLOT,
    SESSION_START_METADATA_SLOT,
    SILENCE_SLOTS,
    SILENCE_TIMEOUT_SLOT,
    SLOT_CONSECUTIVE_SILENCE_TIMEOUTS,
    SLOT_LAST_OBJECT,
    SLOT_LAST_OBJECT_TYPE,
    SLOT_LISTED_ITEMS,
)
from rasa.shared.core.events import (
    AgentCancelled,
    AgentCompleted,
    AgentResumed,
    AgentStarted,
    SlotSet,
)
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.slots import (
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    Slot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import (
    flows_from_str,
)


@pytest.fixture
def mock_available_agents(monkeypatch: MonkeyPatch) -> Iterator[MagicMock]:
    mock_instance = MagicMock()
    mock_instance.agents = {
        "car-research": {},
        "agent-1": {},
        "agent-2": {},
    }

    with patch.object(
        AvailableAgents, "get_instance", return_value=mock_instance
    ) as mock_method:
        # Also patch the step validation to always return True for agent calls
        with patch(
            "rasa.shared.core.flows.steps.call.CallFlowStep.is_calling_agent"
        ) as mock_is_calling:
            mock_is_calling.return_value = True
            yield mock_method


@pytest.fixture
def basic_flow_setup(mock_available_agents):
    """Common setup for agent tests."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )
    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")
    return flows, stack, tracker, flow, step


@pytest.fixture
def agent_stack_frame_setup():
    """Setup for agent stack frame tests."""
    return AgentStackFrame(
        frame_id="test_frame",
        flow_id="test_flow",
        agent_id="test_agent",
        state=AgentState.WAITING_FOR_INPUT,
        metadata={"existing": "data"},
    )


# ============================================================================
# Tests for run_agent function (main integration tests)
# ============================================================================


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_continue_with_user_input(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
    basic_flow_setup,
) -> None:
    flows, stack, tracker, flow, step = basic_flow_setup

    agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="my_flow",
    )
    stack = DialogueStack(frames=[stack.frames[0], agent_stack_frame])
    tracker.update_stack(stack)

    mock_run_agent.return_value = AgentOutput(
        id="test_agent",
        status=AgentStatus.COMPLETED,
        response_message="Agent completed successfully.",
        events=[],
    )

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Assertions
    assert isinstance(flow_step_result, PauseFlowReturnPrediction)
    assert flow_step_result.action_prediction.action_name == ACTION_SEND_TEXT_NAME
    assert mock_run_agent.call_count == 1
    # AgentStackFrame should be removed from stack after successful completion
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_continue_interrupted_agent(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    agent_message = "What is your budget for the car?"
    agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id",
        state=AgentState.INTERRUPTED,
        agent_id="car-research",
        flow_id="my_flow",
        metadata={AGENT_METADATA_AGENT_RESPONSE_KEY: agent_message},
    )
    stack = DialogueStack(frames=[user_stack_frame, agent_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Assertions
    # we expect a PauseFlowReturnPrediction here because the agent was in interrupted
    # state, and we need to re-request user input with the corresponding action
    assert isinstance(flow_step_result, PauseFlowReturnPrediction)
    assert (
        flow_step_result.action_prediction.action_name
        == ACTION_AGENT_REQUEST_USER_INPUT_NAME
    )
    assert (
        flow_step_result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY][
            ACTION_METADATA_TEXT_KEY
        ]
        == agent_message
    )
    # no actual calls to the agent expected
    assert mock_run_agent.call_count == 0
    # AgentStackFrame should be now in WAITING_FOR_INPUT state
    assert isinstance(stack.frames[-1], AgentStackFrame)
    assert cast(AgentStackFrame, stack.frames[-1]).state == AgentState.WAITING_FOR_INPUT
    assert (
        cast(AgentStackFrame, stack.frames[-1]).metadata[
            AGENT_METADATA_AGENT_RESPONSE_KEY
        ]
        == agent_message
    )


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_started(
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )
    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    assert any(
        isinstance(e, AgentStarted) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert isinstance(flow_step_result, ContinueFlowWithNextStep)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_resumed(
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )
    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    # Add an AgentStackFrame simulating an interrupted agent
    agent_stack_frame = AgentStackFrame(
        flow_id="my_flow",
        agent_id="car-research",
        state=AgentState.INTERRUPTED,
        metadata={"agent_response": "Please provide more info"},
    )
    stack = DialogueStack(frames=[user_stack_frame, agent_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    assert any(
        isinstance(e, AgentResumed) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert isinstance(stack.frames[-1], AgentStackFrame)
    assert isinstance(flow_step_result, PauseFlowReturnPrediction)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_completed(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )
    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response: AGENT_COMPLETED
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        events=[],
    )

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    assert any(
        isinstance(e, AgentCompleted) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert isinstance(flow_step_result, ContinueFlowWithNextStep)
    assert mock_run_agent.call_count == 1


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_restart_resets_exit_if_slots_before_agent_call(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    """Test that slots in exit_if conditions are reset before calling the agent."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
              exit_if:
                - slots.amount > 0
                - slots.done is True
                - slots.budget < 50000
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    # Create a restart agent stack frame with the specific frame_id pattern
    restart_agent_stack_frame = AgentStackFrame(
        frame_id="restart_agent_car-research",
        flow_id="my_flow",
        step_id="my-call-step",
        agent_id="car-research",
        state=AgentState.WAITING_FOR_INPUT,
    )
    stack = DialogueStack(frames=[user_stack_frame, restart_agent_stack_frame])
    events = [
        SlotSet("amount", 1000),
        SlotSet("done", True),
        SlotSet("budget", 30000),
        SlotSet("other_slot", "should_not_be_reset"),
    ]
    tracker = DialogueStateTracker.from_events("test", events)
    tracker.update_stack(stack)

    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response to allow run to complete
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        response_message=None,
        events=[],
    )

    await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Verify that the agent was called
    assert mock_run_agent.call_count >= 1

    # Get the agent input that was passed to _call_agent_with_retry
    kwargs = mock_run_agent.call_args.kwargs
    agent_input = kwargs.get("context")
    assert agent_input is not None

    # Verify that slots referenced in exit_if conditions were reset to None
    # before being passed to the agent
    for name in ["amount", "done", "budget"]:
        for slot in agent_input.slots:
            if name == slot.name:
                assert slot.value is None

    # Verify that slots not referenced in exit_if conditions were not affected
    for slot in agent_input.slots:
        if slot.name == "other_slot":
            assert slot.value == "should_not_be_reset"

    # Verify that exit_if conditions are passed to the agent metadata
    assert agent_input.metadata is not None
    assert agent_input.metadata.get("exit_if") == [
        "slots.amount > 0",
        "slots.done is True",
        "slots.budget < 50000",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_status,expected_result_type",
    [
        (AgentStatus.COMPLETED, ContinueFlowWithNextStep),
        (AgentStatus.INPUT_REQUIRED, PauseFlowReturnPrediction),
        (AgentStatus.FATAL_ERROR, ContinueFlowWithNextStep),
    ],
)
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_step_saves_context_id_to_agent_started_event(
    mock_run_agent: AsyncMock,
    agent_status: AgentStatus,
    expected_result_type: type,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    # Mock agent response with context ID
    context_id = "test-context-id"
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=agent_status,
        metadata={A2A_AGENT_CONTEXT_ID_KEY: context_id},
        events=[],
    )

    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Assertions
    assert isinstance(flow_step_result, expected_result_type)
    assert any(
        isinstance(e, AgentStarted) and e.context_id == context_id
        for e in flow_step_result.events
    )


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_context_id_included_in_metadata_when_restarting_agent(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            AgentStarted(
                "car-research", context_id="test-context-id", flow_id="my_flow"
            ),  # The agent was started previously
        ],
    )
    tracker.update_stack(stack)

    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        metadata={},
        events=[],
    )

    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Assertions
    args, kwargs = mock_run_agent.call_args
    assert (
        cast(AgentInput, kwargs["context"]).metadata[A2A_AGENT_CONTEXT_ID_KEY]
        == "test-context-id"
    )


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_fatal_error(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response
    mock_run_agent.return_value = AgentOutput(
        id="car-research", status=AgentStatus.FATAL_ERROR, error_message="Agent failed."
    )

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )

    # Assertions
    assert isinstance(flow_step_result, ContinueFlowWithNextStep)
    # Top frame should be an InternalErrorPatternFlowStackFrame
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)
    # No retries should be made in case of fatal error
    assert mock_run_agent.call_count == 1
    # If the AgentStackFrame was on the stack, it should be removed
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_recoverable_error(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.RECOVERABLE_ERROR,
        error_message="Agent failed.",
        events=[],
    )

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )

    # Assertions
    assert isinstance(flow_step_result, ContinueFlowWithNextStep)
    # Top frame should be an InternalErrorPatternFlowStackFrame
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)
    # Retries should be made in case of recoverable error
    assert mock_run_agent.call_count == MAX_AGENT_RETRIES
    # If the AgentStackFrame was on the stack, it should be removed
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_request_user_input(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response
    agent_message = "What is your budget for the car?"
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.INPUT_REQUIRED,
        response_message=agent_message,
        events=[],
    )

    flow_step_result = await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Assertions
    assert isinstance(flow_step_result, PauseFlowReturnPrediction)
    assert (
        flow_step_result.action_prediction.action_name
        == ACTION_AGENT_REQUEST_USER_INPUT_NAME
    )
    assert (
        flow_step_result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY][
            ACTION_METADATA_TEXT_KEY
        ]
        == agent_message
    )

    assert cast(AgentStackFrame, stack.frames[-1]).state == AgentState.WAITING_FOR_INPUT
    assert (
        cast(AgentStackFrame, stack.frames[-1]).metadata[
            AGENT_METADATA_AGENT_RESPONSE_KEY
        ]
        == agent_message
    )
    assert mock_run_agent.call_count == 1


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_filters_slots_for_agent(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            SlotSet(FLOW_HASHES_SLOT, {"a": 1}),
            SlotSet("keep_me", "value"),
        ],
    )
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response to allow run to complete
    mock_run_agent.return_value = AgentOutput(
        id="test_agent",
        status=AgentStatus.COMPLETED,
        response_message=None,
        events=[],
    )

    await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[TextSlot("keep_me", [])],
    )

    # Verify that slots sent to the agent exclude FLOW_HASHES_SLOT
    assert mock_run_agent.call_count >= 1
    kwargs = mock_run_agent.call_args.kwargs
    agent_input = kwargs.get("context")
    assert agent_input is not None
    assert FLOW_HASHES_SLOT not in agent_input.slots

    for slot in agent_input.slots:
        if slot.name == "keep_me":
            assert slot.value == "value"


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_passes_exit_if_in_metadata(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
              exit_if:
                - slots.amount > 0
                - slots.done is True
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Mock agent response to allow run to complete
    mock_run_agent.return_value = AgentOutput(
        id="test_agent",
        status=AgentStatus.COMPLETED,
        response_message=None,
        events=[],
    )

    await run_agent(
        initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
    )

    # Verify that exit_if made it into the metadata sent to the agent
    assert mock_run_agent.call_count >= 1
    kwargs = mock_run_agent.call_args.kwargs
    agent_input = kwargs.get("context")
    assert agent_input is not None
    assert agent_input.metadata is not None
    assert agent_input.metadata.get("exit_if") == [
        "slots.amount > 0",
        "slots.done is True",
    ]


@pytest.mark.asyncio
async def test_agent_metadata_handling_edge_cases(mock_available_agents):
    """Test agent metadata handling with various edge cases."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: my-call-step
              call: car-research
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    # Test with malformed metadata
    with patch(
        "rasa.core.policies.flows.agent_executor.AgentManager.run_agent"
    ) as mock_run_agent:
        mock_run_agent.return_value = AgentOutput(
            id="car-research",
            status=AgentStatus.COMPLETED,
            metadata={
                "malformed": "data",
                "exit_if": "not_a_list",
            },  # Malformed exit_if
            events=[],
        )

        await run_agent(
            initial_events=[], stack=stack, step=step, tracker=tracker, slots=[]
        )

    # Should not crash and should handle malformed metadata gracefully
    assert mock_run_agent.call_count == 1


@pytest.mark.asyncio
async def test_multiple_agent_calls_in_sequence(mock_available_agents):
    """Test multiple agent calls in sequence."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: first-call
              call: agent-1
            - id: second-call
              call: agent-2
        """
    )

    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    with patch(
        "rasa.core.policies.flows.agent_executor.AgentManager.run_agent"
    ) as mock_run_agent:
        # First agent call
        mock_run_agent.return_value = AgentOutput(
            id="agent-1",
            status=AgentStatus.COMPLETED,
            events=[],
        )

        first_step = flow.step_by_id("first-call")
        result1 = await run_agent(
            initial_events=[], stack=stack, step=first_step, tracker=tracker, slots=[]
        )

        # Second agent call
        mock_run_agent.return_value = AgentOutput(
            id="agent-2",
            status=AgentStatus.COMPLETED,
            events=[],
        )

        second_step = flow.step_by_id("second-call")
        result2 = await run_agent(
            initial_events=[], stack=stack, step=second_step, tracker=tracker, slots=[]
        )

        # Verify both calls were made
        assert mock_run_agent.call_count == 2
        assert isinstance(result1, ContinueFlowWithNextStep)
        assert isinstance(result2, ContinueFlowWithNextStep)


# ============================================================================
# Tests for _prepare_slots_for_agent function
# ============================================================================


@pytest.mark.parametrize(
    "slot_values,slot_definitions,expected_result,exit_if",
    [
        # Test case 1: Empty slots
        ({}, [], [], None),
        # Test case 2: Slots with no definitions
        ({"slot1": "value1", "slot2": "value2"}, [], [], None),
        # Test case 3: Regular slots with definitions
        (
            {"name": "John", "age": 25},
            [TextSlot("name", []), FloatSlot("age", [])],
            [
                AgentInputSlot(
                    name="name",
                    value="John",
                    type="text",
                    allowed_values=None,
                ),
                AgentInputSlot(
                    name="age",
                    value=25,
                    type="float",
                    allowed_values=None,
                ),
            ],
            None,
        ),
        # Test case 4: Categorical slot with allowed values
        (
            {"category": "option1"},
            [CategoricalSlot("category", [], values=["option1", "option2", "option3"])],
            [
                AgentInputSlot(
                    name="category",
                    value="option1",
                    type="categorical",
                    allowed_values=["option1", "option2", "option3"],
                ),
            ],
            None,
        ),
        # Test case 5: Mixed slot types
        (
            {"text": "hello", "number": 42, "category": "A"},
            [
                TextSlot("text", []),
                FloatSlot("number", []),
                CategoricalSlot("category", [], values=["A", "B", "C"]),
            ],
            [
                AgentInputSlot(
                    name="text",
                    value="hello",
                    type="text",
                    allowed_values=None,
                ),
                AgentInputSlot(
                    name="number",
                    value=42,
                    type="float",
                    allowed_values=None,
                ),
                AgentInputSlot(
                    name="category",
                    value="A",
                    type="categorical",
                    allowed_values=["A", "B", "C"],
                ),
            ],
            None,
        ),
        # Test case 6: Slots with excluded slot (FLOW_HASHES_SLOT)
        (
            {"name": "John", FLOW_HASHES_SLOT: "hash123", "age": 25},
            [
                TextSlot("name", []),
                TextSlot(FLOW_HASHES_SLOT, []),
                FloatSlot("age", []),
            ],
            [
                AgentInputSlot(
                    name="name",
                    value="John",
                    type="text",
                    allowed_values=None,
                ),
                AgentInputSlot(
                    name="age",
                    value=25,
                    type="float",
                    allowed_values=None,
                ),
            ],
            None,
        ),
        # Test case 7: Slots with None values
        (
            {"name": None, "age": 0, "active": False},
            [TextSlot("name", []), FloatSlot("age", []), BooleanSlot("active", [])],
            [
                AgentInputSlot(
                    name="age",
                    value=0,
                    type="float",
                    allowed_values=None,
                ),
                AgentInputSlot(
                    name="active",
                    value=False,
                    type="bool",
                    allowed_values=None,
                ),
            ],
            None,
        ),
        # Test case 8: Categorical slot with empty values list
        (
            {"category": "option1"},
            [CategoricalSlot("category", [], values=[])],
            [
                AgentInputSlot(
                    name="category",
                    value="option1",
                    type="categorical",
                    allowed_values=[],
                ),
            ],
            None,
        ),
        # Test case 9: Slots without slot definitions
        (
            {
                "defined_slot": "value1",
                "undefined_slot": "value2",
                "another_defined": "value3",
            },
            [TextSlot("defined_slot", []), TextSlot("another_defined", [])],
            [
                AgentInputSlot(
                    name="defined_slot",
                    value="value1",
                    type="text",
                    allowed_values=None,
                ),
                AgentInputSlot(
                    name="another_defined",
                    value="value3",
                    type="text",
                    allowed_values=None,
                ),
            ],
            None,
        ),
        # Test case 10: Slots with exit_if conditions
        (
            {"name": "John", "age": 25, "confirmation": None},
            [
                TextSlot("name", []),
                FloatSlot("age", []),
                BooleanSlot("confirmation", []),
            ],
            [
                AgentInputSlot(
                    name="name", value="John", type="text", allowed_values=None
                ),
                AgentInputSlot(name="age", value=25, type="float", allowed_values=None),
                AgentInputSlot(
                    name="confirmation", value=None, type="bool", allowed_values=None
                ),
            ],
            ["slots.confirmation is not None", "slots.age > 0"],
        ),
    ],
)
def test_prepare_slots_for_agent(
    slot_values: Dict[str, Any],
    slot_definitions: List[Slot],
    expected_result: List[AgentInputSlot],
    exit_if: Optional[List[str]],
):
    """Test _prepare_slots_for_agent with various input combinations."""
    result = _prepare_slots_for_agent(slot_values, slot_definitions, exit_if)

    assert result == expected_result


def test_prepare_slots_for_agent_excludes_slots_from_slots_excluded_for_agent():
    """Test that SLOTS_EXCLUDED_FOR_AGENT is used to exclude slots."""
    # Create slot values that include all types of excluded slots
    slot_values = {
        # Regular slots that should be included
        "user_name": "John",
        "user_age": 25,
        "user_city": "New York",
        # FLOW_HASHES_SLOT (should be excluded)
        FLOW_HASHES_SLOT: "hash123",
        # DEFAULT_SLOT_NAMES (should be excluded)
        REQUESTED_SLOT: "user_name",
        SESSION_START_METADATA_SLOT: {"key": "value"},
        # SILENCE_SLOTS (should be excluded)
        SILENCE_TIMEOUT_SLOT: 30.0,
        SLOT_CONSECUTIVE_SILENCE_TIMEOUTS: 2,
        # KNOWLEDGE_BASE_SLOT_NAMES (should be excluded)
        SLOT_LISTED_ITEMS: ["item1", "item2"],
        SLOT_LAST_OBJECT: "object123",
        SLOT_LAST_OBJECT_TYPE: "product",
    }

    # Create slot definitions for all slots
    slot_definitions = [
        TextSlot("user_name", []),
        FloatSlot("user_age", []),
        TextSlot("user_city", []),
        TextSlot(FLOW_HASHES_SLOT, []),
        TextSlot(REQUESTED_SLOT, []),
        TextSlot(SESSION_START_METADATA_SLOT, []),
        FloatSlot(SILENCE_TIMEOUT_SLOT, []),
        FloatSlot(SLOT_CONSECUTIVE_SILENCE_TIMEOUTS, []),
        TextSlot(SLOT_LISTED_ITEMS, []),
        TextSlot(SLOT_LAST_OBJECT, []),
        TextSlot(SLOT_LAST_OBJECT_TYPE, []),
    ]

    # Call the function
    result = _prepare_slots_for_agent(slot_values, slot_definitions, None)

    # Verify that only the regular slots are included
    expected_slot_names = {"user_name", "user_age", "user_city"}
    actual_slot_names = {slot.name for slot in result}

    assert actual_slot_names == expected_slot_names

    # Verify that all excluded slots are not present
    excluded_slot_names = {
        FLOW_HASHES_SLOT,
        REQUESTED_SLOT,
        SESSION_START_METADATA_SLOT,
        SILENCE_TIMEOUT_SLOT,
        SLOT_CONSECUTIVE_SILENCE_TIMEOUTS,
        SLOT_LISTED_ITEMS,
        SLOT_LAST_OBJECT,
        SLOT_LAST_OBJECT_TYPE,
    }

    for excluded_slot in excluded_slot_names:
        assert (
            excluded_slot not in actual_slot_names
        ), f"Excluded slot {excluded_slot} should not be in result"

    # Verify the content of the included slots
    assert len(result) == 3

    # Find and verify each included slot
    user_name_slot = next(slot for slot in result if slot.name == "user_name")
    assert user_name_slot.value == "John"
    assert user_name_slot.type == "text"

    user_age_slot = next(slot for slot in result if slot.name == "user_age")
    assert user_age_slot.value == 25
    assert user_age_slot.type == "float"

    user_city_slot = next(slot for slot in result if slot.name == "user_city")
    assert user_city_slot.value == "New York"
    assert user_city_slot.type == "text"

    # Verify that SLOTS_EXCLUDED_FOR_AGENT contains all the expected slots
    expected_excluded_slots = (
        SILENCE_SLOTS | DEFAULT_SLOT_NAMES | KNOWLEDGE_BASE_SLOT_NAMES
    )
    assert SLOTS_EXCLUDED_FOR_AGENT == expected_excluded_slots


def test_reset_slots_covered_by_exit_if():
    """Test _reset_slots_covered_by_exit_if function."""
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            SlotSet("amount", 1000),
            SlotSet("done", True),
            SlotSet("budget", 50000),
            SlotSet("other_slot", "should_not_be_reset"),
        ],
    )

    exit_if_conditions = [
        "slots.amount > 0",
        "slots.done is True",
        "slots.budget < 100000",
    ]

    # Call the function
    _reset_slots_covered_by_exit_if(exit_if_conditions, tracker)

    # Verify that slots referenced in exit_if were reset to None
    assert tracker.get_slot("amount") is None
    assert tracker.get_slot("done") is None
    assert tracker.get_slot("budget") is None

    # Verify that other slots were not affected
    assert tracker.get_slot("other_slot") == "should_not_be_reset"


# ============================================================================
# Tests for _prepare_agent_input function
# ============================================================================


def test_prepare_agent_input_with_exit_if(agent_stack_frame_setup):
    """Test _prepare_agent_input with exit_if conditions."""
    # Create a mock step with exit_if
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        exit_if=["slots.amount > 0", "slots.done is True"],
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    # Create mock tracker with slots
    tracker = DialogueStateTracker.from_events(
        "test",
        [SlotSet("amount", 100), SlotSet("done", True), SlotSet("other", "value")],
    )

    slots = [TextSlot("amount", []), BooleanSlot("done", []), TextSlot("other", [])]

    result = _prepare_agent_input(agent_stack_frame_setup, step, tracker, slots)

    # Verify the result
    assert result.id == "test_agent"
    assert result.user_message == ""
    assert len(result.slots) == 3  # amount, done, other
    assert result.metadata["exit_if"] == ["slots.amount > 0", "slots.done is True"]
    assert result.metadata["existing"] == "data"


def test_prepare_agent_input_without_agent_stack_frame():
    """Test _prepare_agent_input without existing agent stack frame."""
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    tracker = DialogueStateTracker.from_events("test", [SlotSet("amount", 100)])
    slots = [TextSlot("amount", [])]

    result = _prepare_agent_input(None, step, tracker, slots)

    assert result.id == "test_agent"
    assert result.metadata == {}


# ============================================================================
# Tests for remove_agent_stack_frame function
# ============================================================================


def test_remove_agent_stack_frame():
    """Test remove_agent_stack_frame function."""
    # Create a stack with multiple frames including an agent frame
    user_frame = UserFlowStackFrame(flow_id="test_flow", step_id="test_step")
    agent_frame = AgentStackFrame(
        frame_id="agent_frame",
        flow_id="test_flow",
        agent_id="test_agent",
        state=AgentState.WAITING_FOR_INPUT,
    )
    stack = DialogueStack(frames=[user_frame, agent_frame])

    # Verify agent frame exists
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[-1], AgentStackFrame)

    # Remove the agent frame
    remove_agent_stack_frame(stack, "test_agent")

    # Verify agent frame was removed
    assert len(stack.frames) == 1
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


def test_remove_agent_stack_frame_no_agent_frame():
    """Test remove_agent_stack_frame when no agent frame exists."""
    user_frame = UserFlowStackFrame(flow_id="test_flow", step_id="test_step")
    stack = DialogueStack(frames=[user_frame])

    # Should not raise an error
    remove_agent_stack_frame(stack, "nonexistent_agent")

    # Stack should remain unchanged
    assert len(stack.frames) == 1


# ============================================================================
# Tests for action prediction creation functions
# ============================================================================


def test_create_action_prediction():
    """Test _create_action_prediction function."""
    message = "Test message"
    events = [SlotSet("test_slot", "test_value")]

    result = _create_action_prediction(ACTION_SEND_TEXT_NAME, message, events)

    assert result.action_name == ACTION_SEND_TEXT_NAME
    assert (
        result.metadata[ACTION_METADATA_MESSAGE_KEY][ACTION_METADATA_TEXT_KEY]
        == message
    )
    assert result.events == events


def test_create_agent_request_user_input_prediction():
    """Test _create_agent_request_user_input_prediction function."""
    message = "Please provide more information"
    events = [SlotSet("test_slot", "test_value")]

    result = _create_agent_request_user_input_prediction(message, events)

    assert result.action_name == ACTION_AGENT_REQUEST_USER_INPUT_NAME
    assert (
        result.metadata[ACTION_METADATA_MESSAGE_KEY][ACTION_METADATA_TEXT_KEY]
        == message
    )
    assert result.events == events


def test_create_send_text_prediction():
    """Test _create_send_text_prediction function."""
    message = "Hello world"
    events = [SlotSet("test_slot", "test_value")]

    result = _create_send_text_prediction(message, events)

    assert result.action_name == ACTION_SEND_TEXT_NAME
    assert (
        result.metadata[ACTION_METADATA_MESSAGE_KEY][ACTION_METADATA_TEXT_KEY]
        == message
    )
    assert result.events == events


# ============================================================================
# Tests for metadata update functions
# ============================================================================


def test_update_agent_events():
    """Test _update_agent_events function."""
    event = AgentStarted("agent", "flow")
    metadata = {A2A_AGENT_CONTEXT_ID_KEY: "context123"}

    assert event.context_id is None

    _update_agent_events([event], metadata)

    # Verify that the context id was added to the event
    assert event.context_id == "context123"


def test_update_agent_input_metadata_with_events():
    """Test _update_agent_input_metadata_with_events function."""
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            AgentStarted("test_agent", "test_flow", context_id="context123"),
        ],
    )

    metadata = {"existing": "data"}
    agent_id = "test_agent"
    flow_id = "test_flow"

    _update_agent_input_metadata_with_events(metadata, agent_id, flow_id, tracker)

    # Verify that the context id was added to the metadata
    assert A2A_AGENT_CONTEXT_ID_KEY in metadata
    assert metadata[A2A_AGENT_CONTEXT_ID_KEY] == "context123"


# ============================================================================
# Tests for agent status handler functions
# ============================================================================


@pytest.mark.asyncio
async def test_handle_resume_interrupted_agent():
    """Test _handle_resume_interrupted_agent function."""
    # Create mock objects
    agent_stack_frame = AgentStackFrame(
        frame_id="test_frame",
        flow_id="test_flow",
        agent_id="test_agent",
        state=AgentState.INTERRUPTED,
        metadata={AGENT_METADATA_AGENT_RESPONSE_KEY: "Please provide more info"},
    )

    final_events = []
    stack = DialogueStack(frames=[agent_stack_frame])
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )
    tracker = DialogueStateTracker.from_events("test", [])

    result = _handle_resume_interrupted_agent(
        agent_stack_frame, final_events, stack, step, tracker
    )

    # Verify the result
    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == ACTION_AGENT_REQUEST_USER_INPUT_NAME
    assert (
        result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY][
            ACTION_METADATA_TEXT_KEY
        ]
        == "Please provide more info"
    )
    assert isinstance(stack.frames[-1], AgentStackFrame)
    assert cast(AgentStackFrame, stack.frames[-1]).state == AgentState.WAITING_FOR_INPUT


def test_handle_agent_input_required():
    """Test _handle_agent_input_required function."""
    # Create mock objects
    output = AgentOutput(
        id="test_agent",
        status=AgentStatus.INPUT_REQUIRED,
        response_message="What is your budget?",
    )
    final_events = []
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="test_step")]
    )
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    result = _handle_agent_input_required(output, final_events, stack, step)

    # Verify the result
    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == ACTION_AGENT_REQUEST_USER_INPUT_NAME
    assert (
        result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY][
            ACTION_METADATA_TEXT_KEY
        ]
        == "What is your budget?"
    )
    assert isinstance(stack.frames[-1], AgentStackFrame)
    assert cast(AgentStackFrame, stack.frames[-1]).state == AgentState.WAITING_FOR_INPUT
    assert (
        cast(AgentStackFrame, stack.frames[-1]).metadata[
            AGENT_METADATA_AGENT_RESPONSE_KEY
        ]
        == "What is your budget?"
    )


def test_handle_agent_completed():
    """Test _handle_agent_completed function."""
    # Create mock objects
    output = AgentOutput(
        id="test_agent",
        status=AgentStatus.COMPLETED,
        response_message="Task completed successfully",
        events=[SlotSet("result", "success")],
    )
    final_events = []
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="test_step")]
    )
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    result = _handle_agent_completed(output, final_events, stack, step)

    # Verify the result
    assert isinstance(result, PauseFlowReturnPrediction)
    assert any(
        isinstance(e, AgentCompleted) and e.agent_id == "test_agent"
        for e in result.events
    )
    # AgentStackFrame should be removed from stack
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


def test_handle_agent_fatal_error():
    """Test _handle_agent_fatal_error function."""
    # Create mock objects
    output = AgentOutput(
        id="test_agent",
        status=AgentStatus.FATAL_ERROR,
        error_message="Fatal error occurred",
    )
    final_events = []
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="test_step")]
    )
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    result = _handle_agent_fatal_error(output, final_events, stack, step)

    # Verify the result
    assert isinstance(result, ContinueFlowWithNextStep)
    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "test_agent"
        for e in result.events
    )
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)
    # AgentStackFrame should be removed from stack
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


def test_handle_agent_unknown_status():
    """Test handling of unknown agent status."""
    # Create mock objects
    output = AgentOutput(
        id="test_agent",
        status=AgentStatus.FATAL_ERROR,  # Use FATAL_ERROR as a proxy for unknown status
        error_message="Unknown status",
    )
    final_events = []
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="test_flow", step_id="test_step")]
    )
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    result = _handle_agent_unknown_status(output, final_events, stack, step)

    # Verify the result
    assert isinstance(result, ContinueFlowWithNextStep)
    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "test_agent"
        for e in result.events
    )
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)


# ============================================================================
# Tests for _call_agent_with_retry function
# ============================================================================


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_call_agent_with_retry_success(mock_run_agent: AsyncMock):
    """Test _call_agent_with_retry with successful agent call."""
    mock_run_agent.return_value = AgentOutput(
        id="test_agent",
        status=AgentStatus.COMPLETED,
        response_message="Success",
    )

    agent_input = AgentInput(
        id="test_agent",
        user_message="Hello",
        slots=[],
        conversation_history="",
        events=[],
        metadata={},
    )

    result = await _call_agent_with_retry(
        agent_name="test_agent",
        protocol_type=ProtocolType.MCP_OPEN,
        agent_input=agent_input,
        max_retries=3,
    )

    assert result.status == AgentStatus.COMPLETED
    assert mock_run_agent.call_count == 1


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_call_agent_with_retry_recoverable_error(mock_run_agent: AsyncMock):
    """Test _call_agent_with_retry with recoverable error that eventually succeeds."""
    # First call fails with recoverable error, second succeeds
    mock_run_agent.side_effect = [
        AgentOutput(
            id="test_agent",
            status=AgentStatus.RECOVERABLE_ERROR,
            error_message="Temporary failure",
        ),
        AgentOutput(
            id="test_agent",
            status=AgentStatus.COMPLETED,
            response_message="Success after retry",
        ),
    ]

    agent_input = AgentInput(
        id="test_agent",
        user_message="Hello",
        slots=[],
        conversation_history="",
        events=[],
        metadata={},
    )

    result = await _call_agent_with_retry(
        agent_name="test_agent",
        protocol_type=ProtocolType.MCP_TASK,
        agent_input=agent_input,
        max_retries=3,
    )

    assert result.status == AgentStatus.COMPLETED
    assert mock_run_agent.call_count == 2


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_call_agent_with_retry_exception(mock_run_agent: AsyncMock):
    """Test _call_agent_with_retry when agent call raises an exception."""
    mock_run_agent.side_effect = Exception("Network error")

    agent_input = AgentInput(
        id="test_agent",
        user_message="Hello",
        slots=[],
        conversation_history="",
        events=[],
        metadata={},
    )

    result = await _call_agent_with_retry(
        agent_name="test_agent",
        protocol_type=ProtocolType.MCP_OPEN,
        agent_input=agent_input,
        max_retries=3,
    )

    assert result.status == AgentStatus.FATAL_ERROR
    assert mock_run_agent.call_count == 1
