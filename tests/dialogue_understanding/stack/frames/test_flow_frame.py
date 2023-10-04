import pytest
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    InvalidFlowIdException,
    InvalidFlowStackFrameType,
    InvalidFlowStepIdException,
    UserFlowStackFrame,
    FlowStackFrameType,
)
from rasa.shared.core.flows.flow import (
    ActionFlowStep,
    Flow,
    FlowLinks,
    FlowsList,
    StepSequence,
)


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
    flow = Flow(
        id="foo",
        step_sequence=StepSequence(child_steps=[]),
        name="foo flow",
        description="foo flow description",
    )
    all_flows = FlowsList(flows=[flow])
    assert frame.flow(all_flows) == flow


def test_flow_get_flow_non_existant_id():
    frame = UserFlowStackFrame(frame_id="test", flow_id="unknown", step_id="bar")
    all_flows = FlowsList(
        flows=[
            Flow(
                id="foo",
                step_sequence=StepSequence(child_steps=[]),
                name="foo flow",
                description="foo flow description",
            )
        ]
    )
    with pytest.raises(InvalidFlowIdException):
        frame.flow(all_flows)


def test_flow_get_step():
    frame = UserFlowStackFrame(frame_id="test", flow_id="foo", step_id="my_step")
    step = ActionFlowStep(
        idx=1,
        action="action_listen",
        custom_id="my_step",
        description=None,
        metadata={},
        next=FlowLinks(links=[]),
    )
    all_flows = FlowsList(
        flows=[
            Flow(
                id="foo",
                step_sequence=StepSequence(child_steps=[step]),
                name="foo flow",
                description="foo flow description",
            )
        ]
    )
    assert frame.step(all_flows) == step


def test_flow_get_step_non_existant_id():
    frame = UserFlowStackFrame(frame_id="test", flow_id="foo", step_id="unknown")
    all_flows = FlowsList(
        flows=[
            Flow(
                id="foo",
                step_sequence=StepSequence(child_steps=[]),
                name="foo flow",
                description="foo flow description",
            )
        ]
    )
    with pytest.raises(InvalidFlowStepIdException):
        frame.step(all_flows)


def test_flow_get_step_non_existant_flow_id():
    frame = UserFlowStackFrame(frame_id="test", flow_id="unknown", step_id="unknown")
    all_flows = FlowsList(
        flows=[
            Flow(
                id="foo",
                step_sequence=StepSequence(child_steps=[]),
                name="foo flow",
                description="foo flow description",
            )
        ]
    )
    with pytest.raises(InvalidFlowIdException):
        frame.step(all_flows)
