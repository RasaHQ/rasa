import pytest

from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    InvalidFlowIdException,
    InvalidFlowStackFrameType,
    InvalidFlowStepIdException,
    UserFlowStackFrame,
)
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.steps.action import ActionFlowStep


def test_flow_frame_type():
    # types should be stable as they are persisted as part of the tracker
    frame = UserFlowStackFrame(frame_id="test", flow_id="foo", step_id="bar")
    assert frame.type() == "flow"


def test_flow_frame_from_dict():
    frame = UserFlowStackFrame.from_dict(
        {"frame_id": "test", "flow_id": "foo", "step_id": "bar"}
    )
    assert frame.frame_id == "test"
    assert frame.flow_id == "foo"
    assert frame.step_id == "bar"
    assert frame.type() == "flow"
    assert frame.frame_type == FlowStackFrameType.REGULAR


@pytest.mark.parametrize(
    "typ,expected_type",
    [
        ("regular", FlowStackFrameType.REGULAR),
        ("link", FlowStackFrameType.LINK),
        ("interrupt", FlowStackFrameType.INTERRUPT),
        ("call", FlowStackFrameType.CALL),
    ],
)
def test_flow_stack_frame_type_from_str(typ: str, expected_type: FlowStackFrameType):
    assert FlowStackFrameType.from_str(typ) == expected_type


def test_flow_stack_frame_type_from_str_invalid():
    with pytest.raises(InvalidFlowStackFrameType):
        FlowStackFrameType.from_str("invalid")


def test_flow_stack_frame_type_from_str_none():
    assert FlowStackFrameType.from_str(None) == FlowStackFrameType.REGULAR


def test_flow_get_flow():
    frame = UserFlowStackFrame(frame_id="test", flow_id="foo", step_id="bar")
    flow = Flow("foo")
    all_flows = FlowsList([flow])
    assert frame.flow(all_flows) == flow


def test_flow_get_flow_non_existent_id():
    frame = UserFlowStackFrame(frame_id="test", flow_id="unknown", step_id="bar")
    all_flows = FlowsList([Flow("foo")])
    with pytest.raises(InvalidFlowIdException):
        frame.flow(all_flows)


@pytest.mark.parametrize(
    "frame_step_id,flow_step_id",
    (
        # Format of step IDS <=3.11.3
        ("0_collect_step", "0_collect_step"),
        # Format of step IDS >=3.11.4
        ("foo_0_collect_step", "foo_0_collect_step"),
        # Transition from <=3.11.3 to >=3.11.4
        ("0_collect_step", "foo_0_collect_step"),
    ),
)
def test_flow_get_step(frame_step_id: str, flow_step_id: str):
    frame = UserFlowStackFrame(frame_id="test", flow_id="foo", step_id=frame_step_id)
    step = ActionFlowStep(
        idx=1,
        action="action_listen",
        custom_id=flow_step_id,
        description=None,
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="foo",
    )
    all_flows = FlowsList(
        [Flow("foo", step_sequence=FlowStepSequence(child_steps=[step]))]
    )
    assert frame.step(all_flows) == step


def test_flow_get_step_non_existent_id():
    frame = UserFlowStackFrame(frame_id="test", flow_id="foo", step_id="unknown")
    all_flows = FlowsList([Flow("foo")])
    with pytest.raises(InvalidFlowStepIdException):
        frame.step(all_flows)


def test_flow_get_step_non_existent_flow_id():
    frame = UserFlowStackFrame(frame_id="test", flow_id="unknown", step_id="unknown")
    all_flows = FlowsList([Flow("foo")])
    with pytest.raises(InvalidFlowIdException):
        frame.step(all_flows)
