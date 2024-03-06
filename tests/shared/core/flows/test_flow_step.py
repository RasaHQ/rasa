from typing import Type

import pytest

from rasa.shared.core.flows import Flow, FlowStep
from rasa.shared.core.flows.steps import (
    ActionFlowStep,
    CollectInformationFlowStep,
    LinkFlowStep,
    SetSlotsFlowStep,
)
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.flows.steps.no_operation import NoOperationFlowStep
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


@pytest.fixture
def flow_with_all_steps() -> Flow:
    flows = flows_from_str(
        """
            flows:
              test_flow:
                description: test flow
                steps:
                  - id: action_step
                    action: utter_greet
                  - id: set_slots_step
                    set_slots:
                      - has_been_greeted: True
                      - will_be_interesting: unsure
                  - id: collect_step
                    collect: topic
                    ask_before_filling: True
                    reset_after_flow_ends: False
                    rejections:
                      - if: "topic != large language models"
                        utter: utter_too_boring
                  - id: noop_step
                    noop: true
                    next: call_step
                  - id: call_step
                    call: other_flow
                    next: link_step
                  - id: link_step
                    link: other_flow
              other_flow:
                description: other flow
                steps:
                  - id: noop
                    noop: true
                    next: END
        """
    )
    return flows.flow_by_id("test_flow")


@pytest.mark.parametrize(
    "flow_step_id, flow_step_class",
    [
        ("action_step", ActionFlowStep),
        ("set_slots_step", SetSlotsFlowStep),
        ("collect_step", CollectInformationFlowStep),
        ("noop_step", NoOperationFlowStep),
        ("link_step", LinkFlowStep),
    ],
)
def test_flow_step_serialization(
    flow_step_id: str, flow_step_class: Type[FlowStep], flow_with_all_steps: Flow
):
    """Testing for all flow steps that serialization does not add or remove data."""
    step = flow_with_all_steps.step_by_id(flow_step_id)
    assert isinstance(step, flow_step_class)
    step_data = step.as_json()
    step_from_data = flow_step_class.from_json(step_data)
    # overwriting idx of the re-serialized class as this is normally only happening
    # when reading entire flows
    step_from_data.idx = step.idx
    assert step == step_from_data


def test_flow_step_serialization_for_call_step(flow_with_all_steps: Flow):
    # need to test the call step separately, as it contains a reference to
    # another flow, which is not serializable and therefore won't
    # be part of the serialized data.
    step = flow_with_all_steps.step_by_id("call_step")
    assert isinstance(step, CallFlowStep)
    step_data = step.as_json()
    step_from_data = CallFlowStep.from_json(step_data)
    # overwriting idx of the re-serialized class as this is normally only happening
    # when reading entire flows
    step_from_data.idx = step.idx
    assert isinstance(step_from_data, CallFlowStep)
    assert step.call == step_from_data.call


def test_action_flow_step_attributes(flow_with_all_steps: Flow):
    step = flow_with_all_steps.step_by_id("action_step")
    assert isinstance(step, ActionFlowStep)
    assert step.action == "utter_greet"


def test_set_slots_flow_step_attributes(flow_with_all_steps: Flow):
    step = flow_with_all_steps.step_by_id("set_slots_step")
    assert isinstance(step, SetSlotsFlowStep)
    assert len(step.slots) == 2
    assert step.slots[0]["key"] == "has_been_greeted"
    assert step.slots[0]["value"]
    assert step.slots[1]["key"] == "will_be_interesting"
    assert step.slots[1]["value"] == "unsure"


def test_collect_flow_step_attributes(flow_with_all_steps: Flow):
    step = flow_with_all_steps.step_by_id("collect_step")
    assert isinstance(step, CollectInformationFlowStep)
    assert step.collect == "topic"
    assert step.ask_before_filling
    assert not step.reset_after_flow_ends
    assert len(step.rejections) == 1
    assert step.rejections[0].utter == "utter_too_boring"


def test_noop_step_attributes(flow_with_all_steps: Flow):
    step = flow_with_all_steps.step_by_id("noop_step")
    assert isinstance(step, NoOperationFlowStep)
    assert len(step.next.links) == 1
    assert step.next.links[0].target == "call_step"


def test_link_step_attributes(flow_with_all_steps: Flow):
    step = flow_with_all_steps.step_by_id("link_step")
    assert isinstance(step, LinkFlowStep)
    assert step.link == "other_flow"


def test_call_step_attributes(flow_with_all_steps: Flow):
    step = flow_with_all_steps.step_by_id("call_step")
    assert isinstance(step, CallFlowStep)
    assert step.call == "other_flow"


def test_flow_step_always_has_an_id_even_if_not_set():
    step = ActionFlowStep(
        custom_id=None,
        idx=0,
        action="action_listen",
        description=None,
        metadata={},
        next=FlowStepLinks([]),
    )
    assert step.id == "0_action_listen"
    assert step.as_json().get("id") == "0_action_listen"


def test_flow_step_dump_uses_explicit_id():
    step = ActionFlowStep(
        custom_id="foo",
        idx=0,
        action="action_listen",
        description=None,
        metadata={},
        next=FlowStepLinks([]),
    )
    assert step.id == "foo"
    assert step.as_json().get("id") == "foo"
