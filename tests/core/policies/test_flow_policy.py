import textwrap
from typing import List, Optional, Text, Tuple

import pytest

from rasa.core.policies.flow_policy import (
    FlowCircuitBreakerTrippedException,
    FlowExecutor,
    FlowPolicy,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, Event, SlotSet
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
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
            steps:
            - id: "1"
              action: action_listen
              next: "2"
            - id: "2"
              action: action_unlikely_intent   # some action that exists by default
          bar_flow:
            steps:
            - id: first_step
              action: action_listen
        """
    )


def _run_flow_until_listen(
    executor: FlowExecutor, tracker: DialogueStateTracker, domain: Domain
) -> Tuple[List[Optional[Text]], List[Event]]:
    # Run the flow until we reach a listen action.
    # Collect and return all events and intermediate actions.
    events = []
    actions = []
    while True:
        action_prediction = executor.advance_flows(tracker)
        if not action_prediction:
            break

        events.extend(action_prediction.events or [])
        actions.append(action_prediction.action_name)
        tracker.update_with_events(action_prediction.events or [], domain)
        if action_prediction.action_name:
            tracker.update(ActionExecuted(action_prediction.action_name), domain)
        if action_prediction.action_name == "action_listen":
            break
        if action_prediction.action_name is None and not action_prediction.events:
            # No action was executed and no events were generated. This means that
            # the flow isn't doing anything anymore
            break
    return actions, events


def test_select_next_action() -> None:
    flows = YAMLFlowsReader.read_from_string(
        textwrap.dedent(
            """
        flows:
          test_flow:
            description: Test flow
            steps:
              - id: "1"
                intent: transfer_money
                next: "2"
              - id: "2"
                action: utter_ask_name
        """
        )
    )
    tracker = DialogueStateTracker.from_dict(
        "test",
        [
            {"event": "action", "name": "action_listen"},
            {"event": "user", "parse_data": {"intent": {"name": "transfer_money"}}},
        ],
    )
    domain = Domain.empty()
    executor = FlowExecutor.from_tracker(tracker, flows, domain)

    actions, events = _run_flow_until_listen(executor, tracker, domain)

    assert actions == ["flow_test_flow", None]
    assert events == []


def test_flow_policy_does_support_user_flowstack_frame():
    frame = UserFlowStackFrame(flow_id="foo", step_id="first_step", frame_id="some-id")
    assert FlowPolicy.does_support_stack_frame(frame)


def test_flow_policy_does_not_support_search_frame():
    frame = SearchStackFrame(
        frame_id="some-id",
    )
    assert not FlowPolicy.does_support_stack_frame(frame)


def test_get_default_config():
    assert FlowPolicy.get_default_config() == {"priority": 1, "max_history": None}


def test_predict_action_probabilities_abstains_from_unsupported_frame(
    default_flow_policy: FlowPolicy,
):
    domain = Domain.empty()

    stack = DialogueStack(frames=[SearchStackFrame(frame_id="some-id")])
    # create a tracker with the stack set
    tracker = DialogueStateTracker.from_events(
        "test abstain",
        domain=domain,
        slots=domain.slots,
        evts=[ActionExecuted(action_name="action_listen"), stack.persist_as_event()],
    )

    prediction = default_flow_policy.predict_action_probabilities(
        tracker=tracker,
        domain=Domain.empty(),
    )

    # check that the policy didn't predict anything
    assert prediction.max_confidence == 0.0


def test_predict_action_probabilities_advances_topmost_flow(
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
        evts=[ActionExecuted(action_name="action_listen"), stack.persist_as_event()],
    )

    prediction = default_flow_policy.predict_action_probabilities(
        tracker=tracker, domain=Domain.empty(), flows=default_flows
    )

    assert prediction.max_confidence == 1.0

    predicted_idx = prediction.max_confidence_index
    assert domain.action_names_or_texts[predicted_idx] == "action_unlikely_intent"
    # check that the stack was updated
    assert prediction.optional_events == [
        SlotSet(
            "dialogue_stack",
            [
                {
                    "frame_id": "some-id",
                    "flow_id": "foo_flow",
                    "step_id": "2",
                    "frame_type": "regular",
                    "type": "flow",
                }
            ],
        )
    ]


def test_executor_trips_internal_circuit_breaker():
    flow_with_loop = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              set_slot:
                foo: bar
              next: "2"
            - id: "2"
              set_slot:
                foo: barbar
              next: "1"
        """
    )

    domain = Domain.empty()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen"), stack.persist_as_event()],
        domain=domain,
        slots=domain.slots,
    )

    executor = FlowExecutor.from_tracker(tracker, flow_with_loop, domain)

    with pytest.raises(FlowCircuitBreakerTrippedException):
        executor.select_next_action(tracker)


def test_policy_triggers_error_pattern_if_internal_circuit_breaker_is_tripped(
    default_flow_policy: FlowPolicy,
):
    flow_with_loop = flows_from_str_with_defaults(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              set_slot:
                foo: bar
              next: "2"
            - id: "2"
              set_slot:
                foo: barbar
              next: "1"
        """
    )

    domain = flows_default_domain()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen"), stack.persist_as_event()],
        domain=domain,
        slots=domain.slots,
    )

    prediction = default_flow_policy.predict_action_probabilities(
        tracker=tracker, domain=domain, flows=flow_with_loop
    )

    assert prediction.max_confidence == 1.0

    predicted_idx = prediction.max_confidence_index
    assert domain.action_names_or_texts[predicted_idx] == "utter_internal_error_rasa"
    # check that the stack was updated.
    assert len(prediction.optional_events) == 1
    assert isinstance(prediction.optional_events[0], SlotSet)

    assert prediction.optional_events[0].key == "dialogue_stack"
    # the user flow should be on the stack as well as the error pattern
    assert len(prediction.optional_events[0].value) == 2
    # the user flow should be about to end
    assert prediction.optional_events[0].value[0]["step_id"] == "NEXT:END"
    # the pattern should be the other frame
    assert prediction.optional_events[0].value[1]["flow_id"] == "pattern_internal_error"


def test_executor_does_not_get_tripped_if_an_action_is_predicted_in_loop():
    flow_with_loop = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              set_slot:
                foo: bar
              next: "2"
            - id: "2"
              action: action_listen
              next: "1"
        """
    )

    domain = Domain.empty()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen"), stack.persist_as_event()],
        domain=domain,
        slots=domain.slots,
    )

    executor = FlowExecutor.from_tracker(tracker, flow_with_loop, domain)

    selection = executor.select_next_action(tracker)
    assert selection.action_name == "action_listen"
