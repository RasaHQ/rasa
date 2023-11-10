from pytest import CaptureFixture

from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.clarify import (
    ActionClarifyFlows,
    ClarifyPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


async def test_clarify_pattern_flow_stack_frame_type() -> None:
    frame = ClarifyPatternFlowStackFrame()
    assert frame.type() == "pattern_clarification"


async def test_clarify_pattern_flow_stack_frame_from_dict() -> None:
    frame = ClarifyPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            "names": "foo",
            "clarification_options": "",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.names == "foo"
    assert frame.clarification_options == ""
    assert frame.flow_id == "pattern_clarification"
    assert frame.type() == "pattern_clarification"


async def test_action_clarify_flows_assemble_options_string() -> None:
    action = ActionClarifyFlows()

    # empty name
    assert action.assemble_options_string([]) == ""

    # single name
    assert action.assemble_options_string(["option1"]) == "option1"

    # two names
    assert (
        action.assemble_options_string(["option1", "option2"]) == "option1 or option2"
    )

    # multiple names
    assert (
        action.assemble_options_string(["option1", "option2", "option3"])
        == "option1, option2 or option3"
    )


async def test_action_clarify_flows_no_active_flow(capsys: CaptureFixture) -> None:
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionClarifyFlows()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        Domain.empty(),
    )
    assert events == []
    assert "action.clarify_flows.no_active_flow" in capsys.readouterr().out


async def test_action_clarify_flows_no_clarify_frame(capsys: CaptureFixture) -> None:
    domain = Domain.empty()
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[stack.persist_as_event()],
    )
    action = ActionClarifyFlows()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    assert events == []
    assert "action.clarify_flows.no_clarification_frame" in capsys.readouterr().out


async def test_action_clarify_flows() -> None:
    domain = Domain.empty()
    stack = DialogueStack(
        frames=[
            ClarifyPatternFlowStackFrame(
                frame_id="test_id",
                step_id="1",
                names=["foo_flow", "bar_flow"],
                clarification_options="junk",
            )
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[stack.persist_as_event()],
    )
    action = ActionClarifyFlows()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    event = events[0]
    assert isinstance(event, SlotSet)
    assert event.key == DIALOGUE_STACK_SLOT
    assert len(event.value) == 1
    assert event.value[0]["type"] == ClarifyPatternFlowStackFrame.type()
    assert event.value[0]["flow_id"] == "pattern_clarification"
    assert event.value[0]["step_id"] == "1"
    assert event.value[0]["frame_id"] == "test_id"
    assert event.value[0]["names"] == ["foo_flow", "bar_flow"]
    assert event.value[0]["clarification_options"] == "foo_flow or bar_flow"
