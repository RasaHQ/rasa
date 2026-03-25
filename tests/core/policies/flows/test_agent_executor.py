from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import MonkeyPatch

from rasa.agents.constants import (
    A2A_AGENT_CONTEXT_ID_KEY,
    A2A_AGENT_TASK_ID_KEY,
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_RESTARTED_KEY,
    AGENT_METADATA_SENDER_ID_KEY,
)
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.schemas import AgentOutput
from rasa.agents.schemas.agent_input import AgentInput, AgentInputSlot
from rasa.core.constants import (
    ACTIVE_FLOW_METADATA_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY,
    BOT_UTTERANCE_AGENT_NAME_KEY,
    BOT_UTTERANCE_AGENT_TASK_ID_KEY,
    BOT_UTTERANCE_CONTEXT_ID_KEY,
    BOT_UTTERANCE_MESSAGE_ID_KEY,
    STEP_ID_METADATA_KEY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.core.policies.flows.agent_executor import (
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    MAX_AGENT_RETRIES,
    SLOTS_EXCLUDED_FOR_AGENT,
    _build_default_agent_message_metadata,
    _call_agent_with_retry,
    _cancel_flow,
    _create_action_prediction,
    _create_agent_request_user_input_prediction,
    _create_send_text_prediction,
    _handle_agent_cancelled,
    _handle_agent_completed,
    _handle_agent_fatal_error,
    _handle_agent_input_required,
    _handle_agent_unknown_status,
    _prepare_agent_input,
    _prepare_slots_for_agent,
    _reset_slots_covered_by_exit_if,
    _tracker_has_prior_agent_completed,
    _update_agent_events,
    _update_agent_input_metadata_with_events,
    remove_agent_stack_frame,
    run_agent,
)
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    PauseFlowReturnPrediction,
)
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
)
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
    FlowCancelled,
    FlowCompleted,
    SlotSet,
)
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flows_list import FlowsList
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
    # Mock available agents accessed via Configuration singleton
    mock_available_agents_instance = MagicMock()
    mock_available_agents_instance.agents = {
        "car-research": {},
        "agent-1": {},
        "agent-2": {},
    }
    mock_available_agents_instance.get_agent_config.return_value = None

    mock_configuration_instance = MagicMock()
    mock_configuration_instance.available_agents = mock_available_agents_instance

    with patch(
        "rasa.core.config.configuration.Configuration.get_instance",
        return_value=mock_configuration_instance,
    ) as mock_method:
        # Also patch the step validation to always return True for agent calls
        with patch(
            "rasa.shared.core.flows.steps.call.CallFlowStep.is_calling_agent"
        ) as mock_is_calling:
            mock_is_calling.return_value = True
            yield mock_method


@pytest.fixture
def basic_flow_setup(
    mock_available_agents: MagicMock,
) -> Tuple[FlowsList, DialogueStack, DialogueStateTracker, Flow, CallFlowStep]:
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
def agent_stack_frame_setup() -> AgentStackFrame:
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    # Assertions
    assert isinstance(flow_step_result, PauseFlowReturnPrediction)
    assert flow_step_result.action_prediction.action_name == ACTION_SEND_TEXT_NAME
    assert mock_run_agent.call_count == 1
    # AgentStackFrame should be removed from stack after successful completion
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)
    # Ensure flow/step metadata are forwarded to message payload
    payload = flow_step_result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY]
    assert payload[ACTIVE_FLOW_METADATA_KEY] == "my_flow"
    assert payload[STEP_ID_METADATA_KEY] == "my-call-step"


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_continue_interrupted_agent(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    """When agent is INTERRUPTED, we reinvoke the agent with resume metadata."""
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

    # Reinvoked agent returns INPUT_REQUIRED with the same (or new) message
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.INPUT_REQUIRED,
        response_message=agent_message,
    )

    flow_step_result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    # Agent was reinvoked; AgentResumed in events (no AgentStarted when resuming)
    assert any(
        isinstance(e, AgentResumed) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert not any(isinstance(e, AgentStarted) for e in flow_step_result.events)
    assert mock_run_agent.call_count == 1
    # Verify that the agent was called with the correct resume metadata
    context = mock_run_agent.call_args.kwargs["context"]
    metadata = context.metadata
    assert metadata.get("resumed_after_interruption") is True
    assert metadata.get(AGENT_METADATA_AGENT_RESPONSE_KEY) == agent_message
    # Agent returned INPUT_REQUIRED, so we pause for user input
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
    assert isinstance(stack.frames[-1], AgentStackFrame)
    assert cast(AgentStackFrame, stack.frames[-1]).state == AgentState.WAITING_FOR_INPUT


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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert any(
        isinstance(e, AgentStarted) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert isinstance(flow_step_result, ContinueFlowWithNextStep)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_resumed(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    """Resuming an interrupted agent reinvokes it; AgentResumed is in events."""
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

    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.INPUT_REQUIRED,
        response_message="Please provide more info",
    )

    flow_step_result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert any(
        isinstance(e, AgentResumed) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    # AgentStarted is only emitted when starting the agent, not when resuming
    assert not any(isinstance(e, AgentStarted) for e in flow_step_result.events)
    assert mock_run_agent.call_count == 1

    # Ensure the resumed agent is called with the expected resume metadata.
    metadata = mock_run_agent.call_args.kwargs["context"].metadata
    assert metadata.get("resumed_after_interruption") is True
    assert metadata.get(AGENT_METADATA_AGENT_RESPONSE_KEY) == "Please provide more info"
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
    # Restart frame: is_restart=True triggers slot reset and restart metadata
    restart_agent_stack_frame = AgentStackFrame(
        frame_id="restart_agent_car-research",
        flow_id="my_flow",
        step_id="my-call-step",
        agent_id="car-research",
        state=AgentState.WAITING_FOR_INPUT,
        is_restart=True,
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
    # Verify that restarted flag is set when agent is restarted
    assert agent_input.metadata.get(AGENT_METADATA_RESTARTED_KEY) is True


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_reentry_after_completed_resets_exit_if_slots(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    """Prior AgentCompleted + fresh stack entry clears exit_if slots (ENG-2710)."""
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
    slot_defs = [
        FloatSlot("amount", []),
        BooleanSlot("done", []),
        FloatSlot("budget", []),
        TextSlot("other_slot", []),
    ]
    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_stack_frame])
    events = [
        AgentStarted("car-research", "my_flow"),
        AgentCompleted("car-research", "my_flow"),
        SlotSet("amount", 1000),
        SlotSet("done", True),
        SlotSet("budget", 30000),
        SlotSet("other_slot", "should_not_be_reset"),
    ]
    tracker = DialogueStateTracker.from_events("test", events)
    tracker.update_stack(stack)

    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        response_message=None,
        events=[],
    )

    await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=slot_defs,
        flows=flows,
    )

    assert mock_run_agent.call_count >= 1
    agent_input = mock_run_agent.call_args.kwargs["context"]
    assert agent_input is not None
    by_name = {s.name: s.value for s in agent_input.slots}
    assert by_name.get("amount") is None
    assert by_name.get("done") is None
    assert by_name.get("budget") is None
    assert by_name.get("other_slot") == "should_not_be_reset"
    assert agent_input.metadata.get(AGENT_METADATA_RESTARTED_KEY) is True


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_input_required_continuation_does_not_reset_exit_if_slots(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    """Do not reset exit_if slots when continuing INPUT_REQUIRED after a prior run."""
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
        """
    )
    slot_defs = [FloatSlot("amount", []), TextSlot("other_slot", [])]
    user_stack_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="user-frame"
    )
    waiting_frame = AgentStackFrame(
        frame_id="agent-waiting",
        flow_id="my_flow",
        step_id="my-call-step",
        agent_id="car-research",
        state=AgentState.WAITING_FOR_INPUT,
        is_restart=False,
    )
    stack = DialogueStack(frames=[user_stack_frame, waiting_frame])
    events = [
        AgentStarted("car-research", "my_flow"),
        AgentCompleted("car-research", "my_flow"),
        SlotSet("amount", 4242),
        SlotSet("other_slot", "keep_me"),
    ]
    tracker = DialogueStateTracker.from_events("test", events)
    tracker.update_stack(stack)

    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        response_message=None,
        events=[],
    )

    await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=slot_defs,
        flows=flows,
    )

    agent_input = mock_run_agent.call_args.kwargs["context"]
    assert agent_input is not None
    by_name = {s.name: s.value for s in agent_input.slots}
    assert by_name.get("amount") == 4242
    assert by_name.get("other_slot") == "keep_me"
    assert agent_input.metadata.get(AGENT_METADATA_RESTARTED_KEY) is None


def test_tracker_has_prior_agent_completed() -> None:
    """_tracker_has_prior_agent_completed detects completed runs for agent/flow pair."""
    tracker = DialogueStateTracker.from_events(
        "s",
        [
            AgentStarted("a1", "f1"),
            AgentCompleted("a1", "f1"),
        ],
    )
    assert _tracker_has_prior_agent_completed(tracker, "a1", "f1") is True
    assert _tracker_has_prior_agent_completed(tracker, "other", "f1") is False
    assert _tracker_has_prior_agent_completed(tracker, "a1", "other") is False

    empty = DialogueStateTracker.from_events("s", [])
    assert _tracker_has_prior_agent_completed(empty, "a1", "f1") is False


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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert any(
        isinstance(e, FlowCancelled) and e.flow_id == "my_flow"
        for e in flow_step_result.events
    )

    # Assertions
    assert isinstance(flow_step_result, ContinueFlowWithNextStep)
    # Top frame should be an InternalErrorPatternFlowStackFrame
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)
    assert isinstance(stack.frames[-2], CancelPatternFlowStackFrame)
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
async def test_run_agent_completed_forwards_agent_metadata_in_message_payload(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """Ensure metadata is forwarded when COMPLETED returns a response_message."""
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
    agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="my_flow",
    )
    stack = DialogueStack(frames=[user_stack_frame, agent_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    agent_message = "Agent completed with a final response."
    agent_metadata = {
        UTTER_SOURCE_METADATA_KEY: "CustomA2AAgent",
        "custom_key": "custom_value",
    }
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        response_message=agent_message,
        metadata=agent_metadata,
        events=[],
    )

    flow_step_result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    # Should produce a bot message prediction with metadata coming from the agent
    assert isinstance(flow_step_result, PauseFlowReturnPrediction)
    payload = flow_step_result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY]
    assert payload[UTTER_SOURCE_METADATA_KEY] == "CustomA2AAgent"
    # Flow context should also be present
    assert payload[ACTIVE_FLOW_METADATA_KEY] == "my_flow"
    assert payload[STEP_ID_METADATA_KEY] == "my-call-step"


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
        flows=flows,
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
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
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
async def test_agent_metadata_handling_edge_cases(
    mock_available_agents: MagicMock,
) -> None:
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
            initial_events=[],
            stack=stack,
            step=step,
            tracker=tracker,
            slots=[],
            flows=flows,
        )

    # Should not crash and should handle malformed metadata gracefully
    assert mock_run_agent.call_count == 1


@pytest.mark.asyncio
async def test_multiple_agent_calls_in_sequence(
    mock_available_agents: MagicMock,
) -> None:
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
            initial_events=[],
            stack=stack,
            step=first_step,
            tracker=tracker,
            slots=[],
            flows=flows,
        )

        # Second agent call
        mock_run_agent.return_value = AgentOutput(
            id="agent-2",
            status=AgentStatus.COMPLETED,
            events=[],
        )

        second_step = flow.step_by_id("second-call")
        result2 = await run_agent(
            initial_events=[],
            stack=stack,
            step=second_step,
            tracker=tracker,
            slots=[],
            flows=flows,
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
) -> None:
    """Test _prepare_slots_for_agent with various input combinations."""
    result = _prepare_slots_for_agent(slot_values, slot_definitions, exit_if)

    assert result == expected_result


def test_prepare_slots_for_agent_excludes_slots_from_slots_excluded_for_agent() -> None:
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


def test_reset_slots_covered_by_exit_if() -> None:
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


def test_prepare_agent_input_with_exit_if(
    agent_stack_frame_setup: AgentStackFrame,
) -> None:
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
    # Verify tracker metadata is included
    assert result.metadata[AGENT_METADATA_SENDER_ID_KEY] == "test"
    assert result.metadata[AGENT_METADATA_AGENT_ID_KEY] == tracker.assistant_id
    assert result.metadata[AGENT_METADATA_MODEL_ID_KEY] == tracker.model_id


def test_prepare_agent_input_without_agent_stack_frame() -> None:
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
    # Verify tracker metadata is included even when no agent stack frame exists
    assert result.metadata[AGENT_METADATA_SENDER_ID_KEY] == "test"
    assert result.metadata[AGENT_METADATA_AGENT_ID_KEY] == tracker.assistant_id
    assert result.metadata[AGENT_METADATA_MODEL_ID_KEY] == tracker.model_id


def test_prepare_agent_input_events_populated() -> None:
    """Test _prepare_agent_input populates events correctly."""
    from rasa.shared.core.events import BotUttered, SlotSet, UserUttered

    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    # Create tracker with some events
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("i want to book an appointment"),
            BotUttered(
                "to help you book an appointment, please provide further details"
            ),
            SlotSet("booking_complete", "false"),
            UserUttered("next week, mornings, any doctor, can't do wednesdays"),
        ],
    )
    slots = []

    result = _prepare_agent_input(None, step, tracker, slots)

    # Verify that events are populated (not empty)
    assert result.events is not None
    assert len(result.events) > 0

    # Verify that events contain expected event types
    event_types = [event.type_name for event in result.events]
    assert "user" in event_types
    assert "bot" in event_types
    assert "slot" in event_types

    # Verify tracker metadata is included
    assert result.metadata[AGENT_METADATA_SENDER_ID_KEY] == "test"
    assert result.metadata[AGENT_METADATA_AGENT_ID_KEY] == tracker.assistant_id
    assert result.metadata[AGENT_METADATA_MODEL_ID_KEY] == tracker.model_id


def test_prepare_agent_input_tracker_metadata_with_ids() -> None:
    """Test _prepare_agent_input includes assistant_id and model_id."""
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    tracker = DialogueStateTracker.from_events("test_sender", [SlotSet("amount", 100)])
    tracker.assistant_id = "test_assistant_123"
    tracker.model_id = "test_model_456"
    slots = [TextSlot("amount", [])]

    result = _prepare_agent_input(None, step, tracker, slots)

    assert result.id == "test_agent"
    # Verify tracker metadata is correctly set
    assert result.metadata[AGENT_METADATA_SENDER_ID_KEY] == "test_sender"
    assert result.metadata[AGENT_METADATA_AGENT_ID_KEY] == "test_assistant_123"
    assert result.metadata[AGENT_METADATA_MODEL_ID_KEY] == "test_model_456"


# ============================================================================
# Tests for remove_agent_stack_frame function
# ============================================================================


def test_remove_agent_stack_frame() -> None:
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


def test_remove_agent_stack_frame_no_agent_frame() -> None:
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


def test_create_action_prediction() -> None:
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


def test_create_action_prediction_maps_agent_task_id_and_source() -> None:
    """Verify that agent metadata is forwarded into the message payload."""
    message = "Test message"
    events = []
    metadata = {
        A2A_AGENT_TASK_ID_KEY: "task-123",
        UTTER_SOURCE_METADATA_KEY: "A2AAgent",
    }

    result = _create_action_prediction(ACTION_SEND_TEXT_NAME, message, events, metadata)

    payload = result.metadata[ACTION_METADATA_MESSAGE_KEY]
    assert payload[BOT_UTTERANCE_AGENT_TASK_ID_KEY] == "task-123"
    assert payload[UTTER_SOURCE_METADATA_KEY] == "A2AAgent"
    # Original A2A key should not be duplicated inside the payload
    assert A2A_AGENT_TASK_ID_KEY not in payload


def test_create_agent_request_user_input_prediction() -> None:
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


def test_create_send_text_prediction() -> None:
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


def test_update_agent_events() -> None:
    """Test _update_agent_events function."""
    event = AgentStarted("agent", "flow")
    metadata = {A2A_AGENT_CONTEXT_ID_KEY: "context123"}

    assert event.context_id is None

    _update_agent_events([event], metadata)

    # Verify that the context id was added to the event
    assert event.context_id == "context123"


def test_update_agent_input_metadata_with_events() -> None:
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


def test_handle_agent_input_required() -> None:
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


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_input_required_forwards_agent_metadata_in_message_payload(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """Ensure metadata is forwarded when INPUT_REQUIRED returns a response."""
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

    agent_message = "Please provide more details."
    agent_metadata = {UTTER_SOURCE_METADATA_KEY: "CustomTaskAgent", "extra": 123}
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.INPUT_REQUIRED,
        response_message=agent_message,
        metadata=agent_metadata,
        events=[],
    )

    flow_step_result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert isinstance(flow_step_result, PauseFlowReturnPrediction)
    payload = flow_step_result.action_prediction.metadata[ACTION_METADATA_MESSAGE_KEY]
    assert payload[UTTER_SOURCE_METADATA_KEY] == "CustomTaskAgent"
    assert payload[ACTIVE_FLOW_METADATA_KEY] == "my_flow"
    assert payload[STEP_ID_METADATA_KEY] == "my-call-step"


def test_handle_agent_completed() -> None:
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


def test_handle_agent_fatal_error() -> None:
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
    flows = FlowsList([Flow(id="test_flow")])
    tracker = DialogueStateTracker.from_events("test", [])

    result = _handle_agent_fatal_error(
        output, final_events, stack, step, flows, tracker
    )

    # Verify the result
    assert isinstance(result, ContinueFlowWithNextStep)
    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "test_agent"
        for e in result.events
    )
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)
    # AgentStackFrame should be removed from stack
    assert not any(isinstance(frame, AgentStackFrame) for frame in stack.frames)


def test_handle_agent_cancelled() -> None:
    """_handle_agent_cancelled silently ends the flow."""
    output = AgentOutput(
        id="test_agent",
        status=AgentStatus.CANCELLED,
        metadata={"cancellation_reason": "Polling cancelled"},
    )
    final_events = []
    user_frame = UserFlowStackFrame(flow_id="test_flow", step_id="test_step")
    agent_frame = AgentStackFrame(
        frame_id="agent-frame-id",
        flow_id="test_flow",
        agent_id="test_agent",
        state=AgentState.WAITING_FOR_INPUT,
    )
    stack = DialogueStack(frames=[user_frame, agent_frame])
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    result = _handle_agent_cancelled(output, final_events, stack, step)

    assert isinstance(result, ContinueFlowWithNextStep)
    # AgentCancelled with the reason is present
    assert any(
        isinstance(e, AgentCancelled)
        and e.agent_id == "test_agent"
        and e.reason == "Polling cancelled"
        for e in result.events
    )
    # FlowCancelled is present (flow was interrupted, not completed)
    assert any(
        isinstance(e, FlowCancelled) and e.flow_id == "test_flow" for e in result.events
    )
    # No FlowCompleted — the flow was cancelled, not completed
    assert not any(isinstance(e, FlowCompleted) for e in result.events)
    # No error/cancel patterns pushed
    assert not any(
        isinstance(f, InternalErrorPatternFlowStackFrame) for f in stack.frames
    )
    assert not any(isinstance(f, CancelPatternFlowStackFrame) for f in stack.frames)
    # Both agent and flow frames are removed — stack is empty
    assert len(stack.frames) == 0


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_cancelled_is_silent(
    mock_run_agent: AsyncMock,
    monkeypatch: MonkeyPatch,
    mock_available_agents: MagicMock,
) -> None:
    """CANCELLED status emits AgentCancelled only — no error pattern."""
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

    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.CANCELLED,
        metadata={"cancellation_reason": "Streaming cancelled"},
    )

    flow_step_result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert isinstance(flow_step_result, ContinueFlowWithNextStep)
    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "car-research"
        for e in flow_step_result.events
    )
    assert any(
        isinstance(e, FlowCancelled) and e.flow_id == "my_flow"
        for e in flow_step_result.events
    )
    assert not any(isinstance(e, FlowCompleted) for e in flow_step_result.events)
    assert not any(
        isinstance(frame, InternalErrorPatternFlowStackFrame) for frame in stack.frames
    )
    assert not any(
        isinstance(frame, CancelPatternFlowStackFrame) for frame in stack.frames
    )
    # Stack is empty — both agent and flow frames removed
    assert len(stack.frames) == 0
    assert mock_run_agent.call_count == 1


def test_handle_agent_unknown_status() -> None:
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
    flows = FlowsList([Flow(id="test_flow")])
    tracker = DialogueStateTracker.from_events("test", [])

    result = _handle_agent_unknown_status(
        output, final_events, stack, step, flows, tracker
    )

    # Verify the result
    assert isinstance(result, ContinueFlowWithNextStep)
    assert any(
        isinstance(e, AgentCancelled) and e.agent_id == "test_agent"
        for e in result.events
    )
    assert any(
        isinstance(e, FlowCancelled) and e.flow_id == "test_flow" for e in result.events
    )
    assert isinstance(stack.frames[-2], CancelPatternFlowStackFrame)
    assert isinstance(stack.frames[-1], InternalErrorPatternFlowStackFrame)


# ============================================================================
# Tests for _call_agent_with_retry function
# ============================================================================


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_call_agent_with_retry_success(mock_run_agent: AsyncMock) -> None:
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
async def test_call_agent_with_retry_recoverable_error(
    mock_run_agent: AsyncMock,
) -> None:
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
async def test_call_agent_with_retry_exception(mock_run_agent: AsyncMock) -> None:
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


@pytest.mark.parametrize(
    "stack_frames,expected_canceled_name,expected_canceled_frames,expected_flow_id,expected_step_id",
    [
        # Test case 1: UserFlowStackFrame only
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="test_step", frame_id="user-frame-1"
                )
            ],
            "Test flow",
            ["user-frame-1"],
            "test_flow",
            "test_call",
        ),
        # Test case 2: AgentStackFrame on top of UserFlowStackFrame
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="test_step", frame_id="user-frame-1"
                ),
                AgentStackFrame(
                    frame_id="agent-frame-1",
                    flow_id="test_flow",
                    agent_id="test_agent",
                    state=AgentState.WAITING_FOR_INPUT,
                ),
            ],
            "Test flow",
            ["agent-frame-1", "user-frame-1"],
            "test_flow",
            "test_call",
        ),
        # Test case 3: Multiple frames
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="test_step", frame_id="user-frame-2"
                ),
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="test_step", frame_id="user-frame-1"
                ),
                AgentStackFrame(
                    frame_id="agent-frame-1",
                    flow_id="test_flow",
                    agent_id="test_agent",
                    state=AgentState.WAITING_FOR_INPUT,
                ),
            ],
            "Test flow",
            ["agent-frame-1", "user-frame-1"],
            "test_flow",
            "test_call",
        ),
    ],
)
def test_cancel_flow_with_different_stack_configurations(
    stack_frames: List[DialogueStackFrame],
    expected_canceled_name: Optional[str],
    expected_canceled_frames: Optional[List[str]],
    expected_flow_id: Optional[str],
    expected_step_id: Optional[str],
) -> None:
    """Test _cancel_flow with different stack configurations."""
    flows = flows_from_str(
        """
        flows:
          test_flow:
            description: Test flow
            steps:
            - id: test_step
              action: action_listen
        """
    )

    # Create stack with provided frames
    stack = DialogueStack(frames=stack_frames)
    tracker = DialogueStateTracker.from_events("test", [])

    # Create a CallFlowStep
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        metadata={},
    )

    # Call _cancel_flow
    cancel_pattern_frame, flow_cancelled_event = _cancel_flow(
        stack, flows, tracker, step
    )

    # Verify CancelPatternFlowStackFrame
    assert cancel_pattern_frame is not None
    assert isinstance(cancel_pattern_frame, CancelPatternFlowStackFrame)
    assert cancel_pattern_frame.canceled_name.lower() == expected_canceled_name.lower()
    assert cancel_pattern_frame.canceled_frames == expected_canceled_frames

    # Verify FlowCancelled event
    assert flow_cancelled_event is not None
    assert isinstance(flow_cancelled_event, FlowCancelled)
    assert flow_cancelled_event.flow_id == expected_flow_id
    assert flow_cancelled_event.step_id == expected_step_id
    assert flow_cancelled_event.type_name == "flow_cancelled"


@pytest.mark.parametrize(
    "flow_id,flows_config,expected_canceled_name",
    [
        # Test case 1: Existing flow with readable name
        (
            "test_flow",
            """
            flows:
              test_flow:
                description: Test flow
                steps:
                - id: test_step
                  action: action_listen
            """,
            "Test flow",
        ),
        # Test case 2: Non-existent flow (should use flow_id as name)
        (
            "nonexistent_flow",
            """
            flows:
              other_flow:
                description: Other flow
                steps:
                - id: test_step
                  action: action_listen
            """,
            "nonexistent_flow",
        ),
    ],
)
def test_cancel_flow_with_different_flows(
    flow_id: str, flows_config: str, expected_canceled_name: str
) -> None:
    """Test _cancel_flow with different flow configurations."""
    flows = flows_from_str(flows_config)

    # Create a stack with a UserFlowStackFrame
    user_frame = UserFlowStackFrame(
        flow_id=flow_id, step_id="test_step", frame_id="user-frame-1"
    )
    stack = DialogueStack(frames=[user_frame])
    tracker = DialogueStateTracker.from_events("test", [])

    # Create a CallFlowStep
    step = CallFlowStep(
        custom_id="test_call",
        idx=0,
        description="Test call step",
        call="test_agent",
        next=FlowStepLinks(links=[]),
        flow_id=flow_id,
        metadata={},
    )

    # Call _cancel_flow
    cancel_pattern_frame, flow_cancelled_event = _cancel_flow(
        stack, flows, tracker, step
    )

    # Verify results
    assert cancel_pattern_frame is not None
    assert isinstance(cancel_pattern_frame, CancelPatternFlowStackFrame)
    assert cancel_pattern_frame.canceled_name.lower() == expected_canceled_name.lower()
    assert cancel_pattern_frame.canceled_frames == ["user-frame-1"]

    assert flow_cancelled_event is not None
    assert isinstance(flow_cancelled_event, FlowCancelled)
    assert flow_cancelled_event.flow_id == flow_id
    assert flow_cancelled_event.step_id == "test_call"


def test_build_default_agent_message_metadata_with_type() -> None:
    step = SimpleNamespace(call="agent-1", flow_id="flow-1", id="step-1")
    metadata = _build_default_agent_message_metadata(step, "final_response")

    assert metadata[BOT_UTTERANCE_AGENT_NAME_KEY] == "agent-1"
    assert metadata[ACTIVE_FLOW_METADATA_KEY] == "flow-1"
    assert metadata[STEP_ID_METADATA_KEY] == "step-1"
    assert metadata[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] == "final_response"


def test_build_default_agent_message_metadata_without_type() -> None:
    step = SimpleNamespace(call="agent-2", flow_id="flow-2", id="step-2")
    metadata = _build_default_agent_message_metadata(step, None)

    assert metadata[BOT_UTTERANCE_AGENT_NAME_KEY] == "agent-2"
    assert metadata[ACTIVE_FLOW_METADATA_KEY] == "flow-2"
    assert metadata[STEP_ID_METADATA_KEY] == "step-2"
    assert BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY not in metadata


def test_create_action_prediction_maps_all_metadata() -> None:
    # Build rich metadata including protocol IDs and flow/step info
    metadata = {
        UTTER_SOURCE_METADATA_KEY: "CustomA2AAgent",
        BOT_UTTERANCE_AGENT_NAME_KEY: "agent-x",
        A2A_AGENT_TASK_ID_KEY: "task-123",
        A2A_AGENT_CONTEXT_ID_KEY: "ctx-123",
        BOT_UTTERANCE_MESSAGE_ID_KEY: "m-123",
        BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY: "final_response",
        BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY: "2023-10-27T10:00:00Z",
        ACTIVE_FLOW_METADATA_KEY: "flow-1",
        STEP_ID_METADATA_KEY: "step-1",
    }

    pred = _create_action_prediction(
        ACTION_SEND_TEXT_NAME, "hello", events=[], metadata=metadata
    )

    payload = pred.metadata[ACTION_METADATA_MESSAGE_KEY]
    # Text present
    assert payload[ACTION_METADATA_TEXT_KEY] == "hello"
    # Mapped/preserved metadata present
    assert payload[UTTER_SOURCE_METADATA_KEY] == "CustomA2AAgent"
    assert payload[BOT_UTTERANCE_AGENT_NAME_KEY] == "agent-x"
    assert payload[BOT_UTTERANCE_AGENT_TASK_ID_KEY] == "task-123"
    assert payload[BOT_UTTERANCE_CONTEXT_ID_KEY] == "ctx-123"
    assert payload[BOT_UTTERANCE_MESSAGE_ID_KEY] == "m-123"
    assert payload[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] == "final_response"
    assert payload[BOT_UTTERANCE_AGENT_MESSAGE_TIMESTAMP_KEY] == "2023-10-27T10:00:00Z"
    assert payload[ACTIVE_FLOW_METADATA_KEY] == "flow-1"
    assert payload[STEP_ID_METADATA_KEY] == "step-1"
