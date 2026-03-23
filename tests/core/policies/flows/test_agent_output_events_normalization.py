"""Integration-style tests for agent output events (MCP / A2A customizations).

``process_tool_output`` and ``process_agent_output`` must contribute ``Event``
instances (e.g. ``SlotSet``). Serialized dicts used to reach
``attach_stack_metadata_to_events`` and crash with ``AttributeError``; the executor
normalizes them, and these tests lock that behaviour in.
"""

from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.agents.core.types import AgentStatus
from rasa.agents.schemas import AgentOutput
from rasa.core.policies.flows.agent_executor import (
    normalize_agent_output_events,
    run_agent,
)
from rasa.core.policies.flows.flow_executor import attach_stack_metadata_to_events
from rasa.core.policies.flows.flow_step_result import ContinueFlowWithNextStep
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import UserFlowStackFrame
from rasa.shared.core.events import BotUttered, Event, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


def _agent_output_with_raw_events(
    *,
    events: Optional[List[Any]],
    status: AgentStatus = AgentStatus.COMPLETED,
    agent_id: str = "car-research",
) -> AgentOutput:
    """Build ``AgentOutput`` with non-validated ``events``.

    Simulates post-init assignment.
    """
    return AgentOutput.model_construct(
        id=agent_id,
        status=status,
        response_message=None,
        structured_results=None,
        metadata=None,
        timestamp=None,
        error_message=None,
        events=events,
    )


@pytest.fixture
def mock_available_agents(monkeypatch: MonkeyPatch) -> Iterator[MagicMock]:
    mock_available_agents_instance = MagicMock()
    mock_available_agents_instance.agents = {"car-research": {}}
    mock_available_agents_instance.get_agent_config.return_value = None

    mock_configuration_instance = MagicMock()
    mock_configuration_instance.available_agents = mock_available_agents_instance

    with patch(
        "rasa.core.config.configuration.Configuration.get_instance",
        return_value=mock_configuration_instance,
    ) as mock_method:
        with patch(
            "rasa.shared.core.flows.steps.call.CallFlowStep.is_calling_agent"
        ) as mock_is_calling:
            mock_is_calling.return_value = True
            yield mock_method


def test_normalize_agent_output_events_none_and_empty() -> None:
    assert normalize_agent_output_events(None) == []
    assert normalize_agent_output_events([]) == []


def test_normalize_agent_output_events_preserves_slot_set() -> None:
    slot = SlotSet("amount", 42)
    assert slot.metadata == {}
    out = normalize_agent_output_events([slot])
    assert len(out) == 1
    assert out[0] is slot
    assert isinstance(out[0].metadata, dict)


def test_normalize_agent_output_events_converts_slot_dict() -> None:
    raw: Dict[str, Any] = {"event": "slot", "name": "budget", "value": 5000}
    out = normalize_agent_output_events([raw])
    assert len(out) == 1
    assert isinstance(out[0], SlotSet)
    assert out[0].key == "budget"
    assert out[0].value == 5000
    assert isinstance(out[0].metadata, dict)


def test_normalize_agent_output_events_mixed_slot_set_and_dict() -> None:
    s = SlotSet("a", 1)
    d: Dict[str, Any] = {"event": "slot", "name": "b", "value": 2}
    out = normalize_agent_output_events([s, d])
    assert len(out) == 2
    assert out[0] is s
    assert isinstance(out[1], SlotSet) and out[1].key == "b"


def test_normalize_agent_output_events_preserves_bot_uttered() -> None:
    bot = BotUttered(text="Hello", data={}, metadata={"k": "v"})
    out = normalize_agent_output_events([bot])
    assert len(out) == 1
    assert out[0] is bot
    assert isinstance(out[0].metadata, dict)


def test_normalize_agent_output_events_converts_bot_uttered_dict() -> None:
    raw: Dict[str, Any] = {
        "event": "bot",
        "text": "From customization",
        "data": {"buttons": []},
        "metadata": {"custom": True},
    }
    out = normalize_agent_output_events([raw])
    assert len(out) == 1
    assert isinstance(out[0], BotUttered)
    assert out[0].text == "From customization"
    assert out[0].data == {"buttons": []}
    assert out[0].metadata.get("custom") is True


def test_normalize_agent_output_events_mixed_slot_bot_instance_and_dict() -> None:
    """Typical mix: concrete SlotSet, serialized bot line, SlotSet dict."""
    slot_inst = SlotSet("filled", "yes")
    bot_dict: Dict[str, Any] = {"event": "bot", "text": "Ack", "data": {}}
    slot_dict: Dict[str, Any] = {"event": "slot", "name": "other", "value": 3}
    out = normalize_agent_output_events([slot_inst, bot_dict, slot_dict])
    assert len(out) == 3
    assert out[0] is slot_inst
    assert isinstance(out[1], BotUttered) and out[1].text == "Ack"
    assert isinstance(out[2], SlotSet) and out[2].key == "other"


def test_attach_stack_metadata_slot_and_bot_after_normalizing_dicts() -> None:
    events = normalize_agent_output_events(
        [
            {"event": "slot", "name": "x", "value": 1},
            {"event": "bot", "text": "hi", "data": {}},
        ]
    )
    attach_stack_metadata_to_events("step-a", "flow-a", events)
    assert isinstance(events[0], SlotSet)
    assert events[0].metadata["step_id"] == "step-a"
    assert events[0].metadata["active_flow"] == "flow-a"
    assert isinstance(events[1], BotUttered)
    assert events[1].metadata["step_id"] == "step-a"
    assert events[1].metadata["active_flow"] == "flow-a"


def test_attach_stack_metadata_after_normalizing_dict_events() -> None:
    events = normalize_agent_output_events(
        [{"event": "slot", "name": "x", "value": "y"}]
    )
    attach_stack_metadata_to_events("step-1", "flow-1", events)
    assert events[0].metadata["step_id"] == "step-1"
    assert events[0].metadata["active_flow"] == "flow-1"


def test_normalize_agent_output_events_logs_error_for_non_event_non_dict() -> None:
    with capture_logs() as cap:
        out = normalize_agent_output_events(
            [SlotSet("ok", 1), "not-a-valid-event"], agent_name="my-agent"
        )
    assert len(out) == 1
    assert out[0].key == "ok"
    errors = [e for e in cap if e.get("log_level") == "error"]
    assert any(
        e.get("event")
        == "flow_executor.normalize_agent_output_events.unsupported_event_type"
        and e.get("agent_name") == "my-agent"
        and e.get("unsupported_event_type") == "str"
        for e in errors
    )


def test_normalize_agent_output_events_logs_error_for_unknown_dict_event_type() -> None:
    with capture_logs() as cap:
        out = normalize_agent_output_events(
            [{"event": "totally_unknown_xyz", "foo": 1}], agent_name="a2a"
        )
    assert out == []
    errors = [e for e in cap if e.get("log_level") == "error"]
    # Unknown ``event`` names raise from ``resolve_by_type`` → ``parse_failed`` log.
    assert any(
        e.get("event") == "flow_executor.normalize_agent_output_events.parse_failed"
        and e.get("agent_name") == "a2a"
        and e.get("declared_type") == "totally_unknown_xyz"
        and e.get("parameter_keys") == ["event", "foo"]
        for e in errors
    )


def test_normalize_agent_output_events_logs_error_when_dict_missing_event_key() -> None:
    with capture_logs() as cap:
        out = normalize_agent_output_events([{"name": "only_name", "value": 1}])
    assert out == []
    errors = [e for e in cap if e.get("log_level") == "error"]
    assert any(
        e.get("event")
        == "flow_executor.normalize_agent_output_events.unsupported_dict_event"
        and e.get("declared_type") is None
        and e.get("parameter_keys") == ["name", "value"]
        for e in errors
    )


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_normalizes_dict_slot_events_before_flow_handlers(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """AgentOutput.events as slot dicts must not break completed-agent handling."""
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
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="my_flow", step_id="START", frame_id="f1")]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = _agent_output_with_raw_events(
        events=[{"event": "slot", "name": "from_agent", "value": 1}],
    )

    result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert result.events
    slot_events = [e for e in result.events if isinstance(e, SlotSet)]
    assert any(e.key == "from_agent" and e.value == 1 for e in slot_events)
    for e in result.events:
        if isinstance(e, SlotSet):
            assert isinstance(e.metadata, dict)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_dict_events_compatible_with_attach_stack_metadata(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """Same path the policy uses: normalized events accept stack metadata writes."""
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
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="my_flow", step_id="START", frame_id="f1")]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = _agent_output_with_raw_events(
        events=[{"event": "slot", "name": "s", "value": True}],
    )

    result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    events: List[Event] = list(result.events)
    attach_stack_metadata_to_events(step.id, flow.id, events)
    for e in events:
        if isinstance(e, SlotSet) and e.key == "s":
            assert "step_id" in e.metadata
            assert "active_flow" in e.metadata
            return
    pytest.fail("Expected SlotSet from normalized dict in result.events")


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
@pytest.mark.parametrize("events_field", [None, []])
async def test_run_agent_completed_with_no_slot_events(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
    events_field: Optional[List[Any]],
) -> None:
    """Agent may return no events (None or empty list) on COMPLETED."""
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
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="my_flow", step_id="START", frame_id="f1")]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        events=events_field,
    )

    result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events is not None
    assert not any(isinstance(e, SlotSet) for e in result.events)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_preserves_bot_uttered_instance(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """Agent may return BotUttered objects directly (not serialized dicts)."""
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
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="my_flow", step_id="START", frame_id="f1")]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    utterance = BotUttered(text="Already an event", data={})
    mock_run_agent.return_value = AgentOutput(
        id="car-research",
        status=AgentStatus.COMPLETED,
        events=[utterance],
    )

    result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    assert utterance in result.events


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_normalizes_bot_uttered_dict(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """Agent output may carry BotUttered as serialized dicts."""
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
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="my_flow", step_id="START", frame_id="f1")]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = _agent_output_with_raw_events(
        events=[
            {"event": "bot", "text": "Tool finished", "data": {}},
        ],
    )

    result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    bot_events = [e for e in result.events if isinstance(e, BotUttered)]
    assert len(bot_events) == 1
    assert bot_events[0].text == "Tool finished"
    assert isinstance(bot_events[0].metadata, dict)


@pytest.mark.asyncio
@patch("rasa.core.policies.flows.agent_executor.AgentManager.run_agent")
async def test_run_agent_normalizes_slot_and_bot_dict_together(
    mock_run_agent: AsyncMock,
    mock_available_agents: MagicMock,
) -> None:
    """Agent output can combine SlotSet dicts and BotUttered dicts."""
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
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="my_flow", step_id="START", frame_id="f1")]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("my-call-step")

    mock_run_agent.return_value = _agent_output_with_raw_events(
        events=[
            {"event": "slot", "name": "status", "value": "ok"},
            {"event": "bot", "text": "Done", "data": {}},
        ],
    )

    result = await run_agent(
        initial_events=[],
        stack=stack,
        step=step,
        tracker=tracker,
        slots=[],
        flows=flows,
    )

    events_list: List[Event] = list(result.events)
    attach_stack_metadata_to_events(step.id, flow.id, events_list)

    slots = [e for e in events_list if isinstance(e, SlotSet)]
    bots = [e for e in events_list if isinstance(e, BotUttered)]
    assert any(s.key == "status" and s.value == "ok" for s in slots)
    assert any(b.text == "Done" for b in bots)
    for e in slots + bots:
        assert "step_id" in e.metadata
        assert "active_flow" in e.metadata
