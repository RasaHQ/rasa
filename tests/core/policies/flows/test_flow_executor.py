import textwrap
from typing import List, Optional, Tuple
import pytest
from rasa.core.policies.flows import flow_executor
from rasa.core.policies.flows.flow_exceptions import FlowCircuitBreakerTrippedException
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import UserFlowStackFrame
from rasa.dialogue_understanding.stack.frames.search_frame import SearchStackFrame
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, Event, SlotSet
from rasa.shared.core.flows.flow import (
    END_STEP,
    ContinueFlowStep,
    EndFlowStep,
    FlowLinks,
    FlowsList,
    SetSlotsFlowStep,
)
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader, flows_from_str
from rasa.shared.core.slots import TextSlot
from rasa.shared.core.trackers import DialogueStateTracker


def test_render_template_variables():
    assert (
        flow_executor.render_template_variables("foo {{bar}}", {"bar": "bar baz"})
        == "foo bar baz"
    )


def test_render_template_empty_context():
    assert flow_executor.render_template_variables("foo {{bar}}", {}) == "foo "


def test_render_template_empty_text():
    assert flow_executor.render_template_variables("", {"bar": "bar baz"}) == ""


def test_evaluate_simple_predicate():
    predicate = "2 > 1"
    stack = DialogueStack(frames=[])
    tracker = DialogueStateTracker.from_events("test", [])
    assert flow_executor.is_condition_satisfied(predicate, stack, tracker)


def test_evaluate_simple_predicate_failing():
    predicate = "2 < 1"
    stack = DialogueStack(frames=[])
    tracker = DialogueStateTracker.from_events("test", [])
    assert not flow_executor.is_condition_satisfied(predicate, stack, tracker)


def test_invalid_predicate():
    predicate = "2 >!= 1"
    stack = DialogueStack(frames=[])
    tracker = DialogueStateTracker.from_events("test", [])
    assert not flow_executor.is_condition_satisfied(predicate, stack, tracker)


def test_evaluate_predicate_with_context_unsuccessfully():
    predicate = "'foo' = context.flow_id"
    unsatisfied_tracker = DialogueStateTracker.from_events("test", [])
    assert not flow_executor.is_condition_satisfied(
        predicate,
        context={},
        tracker=unsatisfied_tracker,
    )


def test_evaluate_predicate_with_context_successfully():
    predicate = "'foo' = context.flow_id"
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            )
        ]
    )

    satisfied_tracker = DialogueStateTracker.from_events(
        "test", [stack.persist_as_event()]
    )
    assert flow_executor.is_condition_satisfied(
        predicate,
        stack.current_context(),
        satisfied_tracker,
    )


def test_evaluate_predicate_with_slots():
    predicate = "'foo' = my_slot"

    satisfied_tracker = DialogueStateTracker.from_events(
        "test", [SlotSet("my_slot", "foo")]
    )
    assert flow_executor.is_condition_satisfied(
        predicate,
        context={},
        tracker=satisfied_tracker,
    )

    unsatisfied_tracker = DialogueStateTracker.from_events("test", [])
    assert not flow_executor.is_condition_satisfied(
        predicate,
        context={},
        tracker=unsatisfied_tracker,
    )


def test_is_step_end_of_flow_is_false_for_set_slot():
    step = SetSlotsFlowStep(
        custom_id="foo",
        description="",
        idx=1,
        slots=[],
        metadata={},
        next=FlowLinks(links=[]),
    )
    assert not flow_executor.is_step_end_of_flow(step)


def test_is_step_end_of_flow_is_true_for_end():
    step = EndFlowStep()
    assert flow_executor.is_step_end_of_flow(step)


def test_is_step_end_of_flow_is_true_for_step_continuing_at_end():
    step = ContinueFlowStep(next=END_STEP)
    assert flow_executor.is_step_end_of_flow(step)


def test_select_next_step_static_link():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [stack.persist_as_event()])
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == "collect_bar"
    )


def test_select_next_step_branch_if():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
              next:
              - if: foo is 'foobar'
                then: collect_bar
              - else:
                - id: collect_baz
                  collect: baz
                  next: END
            - id: collect_bar
              collect: bar
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events(
        "test", [stack.persist_as_event(), SlotSet("foo", "foobar")]
    )
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == "collect_bar"
    )


def test_select_next_step_branch_else():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
              next:
              - if: foo is 'foobar'
                then: collect_bar
              - else:
                - id: collect_baz
                  collect: baz
                  next: END
            - id: collect_bar
              collect: bar
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events(
        "test", [stack.persist_as_event(), SlotSet("foo", "bazbaz")]
    )
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == "collect_baz"
    )


def test_select_next_step_branch_not_possible():
    # the flow is missing an else so we can't select a next step
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
              next:
              - if: foo is 'foobar'
                then: collect_bar
              - if: foo is 'fooooobar'
                then:
                - id: collect_baz
                  collect: baz
                  next: END
              # we need to add this when parsing, otherwise it fails. but
              # we will remove it later in the test code
              - else: END
            - id: collect_bar
              collect: bar
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events(
        "test", [stack.persist_as_event(), SlotSet("foo", "bazbaz")]
    )
    step = user_flow_frame.step(all_flows)

    step.next.links = step.next.links[:-1]  # removes the else branch

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        is None
    )


def test_select_handles_END_next():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [stack.persist_as_event()])
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == "END"
    )


def test_select_handles_no_next():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [stack.persist_as_event()])
    step = user_flow_frame.step(all_flows)

    # we need to manually create this case as the YAML parser doesn't allow
    # for empty nexts. so actually, we shouldn't even get into this situation
    # but still good to make sure that the function handles it
    step.next = FlowLinks(links=[])

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        is None
    )


def test_select_handles_current_node_being_END():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="END", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [stack.persist_as_event()])
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        is None
    )


def test_select_handles_current_node_being_link():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            steps:
            - id: link_to_foo
              link: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="link_to_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [stack.persist_as_event()])
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == "END"
    )


def test_advance_top_flow_on_stack_handles_empty_stack():
    stack = DialogueStack(frames=[])
    flow_executor.advance_top_flow_on_stack("foo", stack)
    assert stack == DialogueStack(frames=[])


def test_advance_top_flow_on_stack_handles_non_user_flow_stack():
    search_frame = SearchStackFrame(frame_id="some-frame-id")
    stack = DialogueStack(frames=[search_frame])
    flow_executor.advance_top_flow_on_stack("foo", stack)
    assert stack == DialogueStack(frames=[search_frame])


def test_advance_top_flow_on_stack_advances_user_flow():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    flow_executor.advance_top_flow_on_stack("bar", stack)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.step_id == "bar"


def test_executor_does_not_get_tripped_if_an_action_is_predicted_in_loop():
    flow_with_loop = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              set_slots:
                - foo: bar
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

    selection = flow_executor.select_next_action(stack, tracker, domain, flow_with_loop)
    assert selection.action_name == "action_listen"


def test_resets_all_slots_after_flow_ends() -> None:
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              collect: my_slot
            - id: "2"
              set_slots:
                - foo: bar
                - other_slot: other_value
            - id: "3"
              action: action_listen
        """
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            SlotSet("my_slot", "my_value"),
            SlotSet("foo", "bar"),
            SlotSet("other_slot", "other_value"),
            ActionExecuted("action_listen"),
        ],
        slots=[
            TextSlot("my_slot", mappings=[], initial_value="initial_value"),
            TextSlot("foo", mappings=[]),
            TextSlot("other_slot", mappings=[]),
        ],
    )

    current_flow = flows.flow_by_id("foo_flow")
    events = flow_executor.reset_scoped_slots(current_flow, tracker)
    assert events == [
        SlotSet("my_slot", "initial_value"),
        SlotSet("foo", None),
        SlotSet("other_slot", None),
    ]


def test_set_slots_inherit_reset_from_collect_step() -> None:
    """Test that `reset_after_flow_ends` is inherited from the collect step."""
    slot_name = "my_slot"
    flows = flows_from_str(
        f"""
        flows:
          foo_flow:
            steps:
            - id: "1"
              collect: {slot_name}
              reset_after_flow_ends: false
            - id: "2"
              set_slots:
                - foo: bar
                - {slot_name}: my_value
            - id: "3"
              action: action_listen
        """
    )
    tracker = DialogueStateTracker.from_events(
        "test123",
        [
            SlotSet("my_slot", "my_value"),
            SlotSet("foo", "bar"),
            ActionExecuted("action_listen"),
        ],
        slots=[
            TextSlot("my_slot", mappings=[], initial_value="initial_value"),
            TextSlot("foo", mappings=[]),
        ],
    )

    current_flow = flows.flow_by_id("foo_flow")
    events = flow_executor.reset_scoped_slots(current_flow, tracker)
    assert events == [
        SlotSet("foo", None),
    ]


def test_executor_trips_internal_circuit_breaker():
    flow_with_loop = flows_from_str(
        """
        flows:
          foo_flow:
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

    with pytest.raises(FlowCircuitBreakerTrippedException):
        flow_executor.select_next_action(stack, tracker, domain, flow_with_loop)


def _run_flow_until_listen(
    tracker: DialogueStateTracker, domain: Domain, flows: FlowsList
) -> Tuple[List[Optional[str]], List[Event]]:
    # Run the flow until we reach a listen action.
    # Collect and return all events and intermediate actions.
    events = []
    actions = []
    while True:
        action_prediction = flow_executor.advance_flows(tracker, domain, flows)
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


@pytest.mark.skip(reason="Skip until intent gets replaced by nlu_trigger")
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

    actions, events = _run_flow_until_listen(tracker, domain, flows)

    assert actions == ["flow_test_flow", None]
    assert events == []
