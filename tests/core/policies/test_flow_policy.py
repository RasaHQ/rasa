from typing import Optional

import pytest

from rasa.core.policies.flow_policy import (
    FlowPolicy,
)
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    DialogueStackUpdated,
    FlowStarted,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.dialogue_understanding.stack.frames import (
    UserFlowStackFrame,
    SearchStackFrame,
)
from tests.utilities import (
    flows_default_domain,
    flows_from_str,
    flows_from_str_with_defaults,
)


@pytest.fixture()
def resource() -> Resource:
    return Resource("flow_policy")


@pytest.fixture()
def default_flow_policy(
    resource: Resource,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> FlowPolicy:
    return FlowPolicy(
        config={},
        model_storage=default_model_storage,
        resource=resource,
        execution_context=default_execution_context,
    )


@pytest.fixture()
def default_flows() -> FlowsList:
    return flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              action: action_listen
              next: "2"
            - id: "2"
              action: action_unlikely_intent   # some action that exists by default
          bar_flow:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )


def test_flow_policy_does_support_user_flowstack_frame():
    frame = UserFlowStackFrame(flow_id="foo", step_id="first_step", frame_id="some-id")
    assert FlowPolicy.does_support_stack_frame(frame)


def test_flow_policy_does_not_support_search_frame():
    frame = SearchStackFrame(
        frame_id="some-id",
    )
    assert not FlowPolicy.does_support_stack_frame(frame)


def test_get_default_config():
    assert FlowPolicy.get_default_config() == {"priority": 7, "max_history": None}


async def test_predict_action_probabilities_abstains_in_coexistence(
    default_flow_policy: FlowPolicy, default_flows: FlowsList
):
    domain = Domain.empty()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )
    # create a tracker with the stack set
    tracker = DialogueStateTracker.from_events(
        "test abstain",
        domain=domain,
        slots=domain.slots
        + [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[{}], initial_value=False)],
        evts=[ActionExecuted(action_name="action_listen")],
    )
    tracker.update_stack(stack)

    prediction = await default_flow_policy.predict_action_probabilities(
        tracker=tracker, domain=Domain.empty(), flows=default_flows
    )

    # check that the policy didn't predict anything
    assert prediction.max_confidence == 0.0


async def test_predict_action_probabilities_abstains_from_unsupported_frame(
    default_flow_policy: FlowPolicy,
):
    domain = Domain.empty()

    stack = DialogueStack(frames=[SearchStackFrame(frame_id="some-id")])
    # create a tracker with the stack set
    tracker = DialogueStateTracker.from_events(
        "test abstain",
        domain=domain,
        slots=domain.slots,
        evts=[ActionExecuted(action_name="action_listen")],
    )
    tracker.update_stack(stack)

    prediction = await default_flow_policy.predict_action_probabilities(
        tracker=tracker,
        domain=Domain.empty(),
    )

    # check that the policy didn't predict anything
    assert prediction.max_confidence == 0.0


async def test_predict_action_probabilities_advances_topmost_flow(
    default_flow_policy: FlowPolicy, default_flows: FlowsList
):
    domain = Domain.empty()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )

    tracker = DialogueStateTracker.from_events(
        "test abstain",
        domain=domain,
        slots=domain.slots,
        evts=[ActionExecuted(action_name="action_listen")],
    )
    tracker.update_stack(stack)

    prediction = await default_flow_policy.predict_action_probabilities(
        tracker=tracker, domain=Domain.empty(), flows=default_flows
    )

    assert prediction.max_confidence == 1.0

    predicted_idx = prediction.max_confidence_index
    assert domain.action_names_or_texts[predicted_idx] == "action_unlikely_intent"
    # check that the stack was updated
    assert len(prediction.optional_events) == 1
    dialogue_stack_event = prediction.optional_events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "2"
    assert frame.frame_id is not None
    assert frame.frame_type == "regular"


async def test_policy_triggers_error_pattern_if_internal_circuit_breaker_is_tripped(
    default_flow_policy: FlowPolicy,
):
    flow_with_loop = flows_from_str_with_defaults(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              set_slots:
                - foo: bar
              next: "2"
            - id: "2"
              set_slots:
                - foo: barbar
              next: "1"
        """
    )

    domain = flows_default_domain()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen")],
        domain=domain,
        slots=domain.slots,
    )
    tracker.update_stack(stack)

    prediction = await default_flow_policy.predict_action_probabilities(
        tracker=tracker.copy(), domain=domain, flows=flow_with_loop
    )

    assert prediction.max_confidence == 1.0

    predicted_idx = prediction.max_confidence_index
    assert domain.action_names_or_texts[predicted_idx] == "utter_internal_error_rasa"
    # check that the stack was updated.
    assert prediction.optional_events[2] == FlowStarted(
        flow_id="pattern_internal_error"
    )

    assert isinstance(prediction.optional_events[0], DialogueStackUpdated)
    assert isinstance(prediction.optional_events[1], DialogueStackUpdated)

    tracker.update_with_events(prediction.optional_events)
    updated_stack = tracker.stack

    # the user flow should be on the stack as well as the error pattern
    assert len(updated_stack.frames) == 2

    # the user flow should be about to end
    assert isinstance(updated_stack.frames[0], UserFlowStackFrame)
    assert updated_stack.frames[0].step_id == "NEXT:END"
    # the pattern should be the other frame
    assert isinstance(updated_stack.frames[1], InternalErrorPatternFlowStackFrame)


async def test_policy_cancels_user_flow_and_trigger_error_pattern_invalid_custom_slots(
    default_flow_policy: FlowPolicy,
):
    test_flows = flows_from_str_with_defaults(
        """
        flows:
          foo_flow:
            description: flow foo
            nlu_trigger:
             - intent: foo_intent
            steps:
            - collect: bar_slot
              next: END
        """
    )

    domain = Domain.from_yaml("""
    intents:
    - foo_intent
    slots:
        bar_slot:
          type: text
          mappings:
          - type: custom
    """).merge(flows_default_domain())

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[UserUttered("Hi", intent={"name": "foo_intent"})],
        domain=domain,
        slots=domain.slots,
    )

    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(flow_id="foo_flow", step_id="START", frame_id="some-id")
        ]
    )
    tracker.update_stack(stack)

    prediction = await default_flow_policy.predict_action_probabilities(
        tracker=tracker.copy(), domain=domain, flows=test_flows
    )

    assert prediction.max_confidence == 1.0

    predicted_idx = prediction.max_confidence_index
    assert domain.action_names_or_texts[predicted_idx] == "utter_internal_error_rasa"
    # check that the stack was updated.
    assert isinstance(prediction.optional_events[0], DialogueStackUpdated)
    assert isinstance(prediction.optional_events[1], DialogueStackUpdated)
    assert prediction.optional_events[2] == FlowStarted(flow_id="foo_flow")
    assert isinstance(prediction.optional_events[3], DialogueStackUpdated)
    assert prediction.optional_events[4] == FlowStarted(
        flow_id="pattern_internal_error"
    )
    assert isinstance(prediction.optional_events[5], DialogueStackUpdated)

    tracker.update_with_events(prediction.optional_events)
    updated_stack = tracker.stack

    # the user flow, the cancel flow and error patterns should be all on the stack
    assert len(updated_stack.frames) == 3

    first_frame = updated_stack.frames[0]
    assert isinstance(first_frame, UserFlowStackFrame)
    assert first_frame.flow_id == "foo_flow"
    assert first_frame.step_id == "0_collect_bar_slot"
    second_frame = updated_stack.frames[1]
    assert isinstance(second_frame, CancelPatternFlowStackFrame)
    assert second_frame.canceled_name == "foo_flow"
    assert second_frame.canceled_frames == ["some-id"]
    # the error pattern should be the other frame
    assert isinstance(updated_stack.frames[2], InternalErrorPatternFlowStackFrame)


@pytest.mark.parametrize(
    "routing_slot_value,result",
    [
        (None, True),
        (True, False),
        (False, True),
    ],
)
def test_should_abstain_in_coexistence(
    routing_slot_value: Optional[bool], result: bool, default_flow_policy: FlowPolicy
):
    tracker = DialogueStateTracker(
        "id1",
        slots=[BooleanSlot(ROUTE_TO_CALM_SLOT, [], initial_value=routing_slot_value)],
    )

    assert result == default_flow_policy.should_abstain_in_coexistence(tracker, True)
