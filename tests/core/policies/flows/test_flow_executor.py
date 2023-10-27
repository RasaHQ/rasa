import textwrap
from typing import List, Optional, Tuple
import pytest
from rasa.core.policies.flows import flow_executor
from rasa.core.policies.flows.flow_exceptions import FlowCircuitBreakerTrippedException
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.completed import (
    CompletedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.chit_chat_frame import ChitChatStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
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
    SlotRejection,
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


def test_trigger_pattern_continue_interrupted_adds_stackframe():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            name: bar flow
            steps:
            - id: "2"
              collect: bar
        """
    )

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="bar_flow", step_id="2", frame_id="some-id")]
    )

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow",
        step_id="1",
        frame_id="some-other-id",
        frame_type=FlowStackFrameType.INTERRUPT,
    )

    flow_executor.trigger_pattern_continue_interrupted(current_frame, stack, flows)

    top = stack.top()
    assert top is not None
    assert isinstance(top, ContinueInterruptedPatternFlowStackFrame)
    assert top.previous_flow_name == "bar flow"


def test_trigger_pattern_continue_interrupted_does_not_trigger_if_no_interrupt():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            name: bar flow
            steps:
            - id: "2"
              collect: bar
        """
    )

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="bar_flow", step_id="2", frame_id="some-id")]
    )

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(current_frame, stack, flows)

    # only the original frame should be on the stack
    assert len(stack.frames) == 1


def test_trigger_pattern_continue_interrupted_does_not_trigger_if_frame_is_already_finished():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            name: bar flow
            steps:
            - id: "2"
              collect: bar
        """
    )

    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="bar_flow",
                step_id="END",
                frame_id="some-id",
                frame_type=FlowStackFrameType.INTERRUPT,
            )
        ]
    )

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(current_frame, stack, flows)

    # only the original frame should be on the stack
    assert len(stack.frames) == 1


def test_trigger_pattern_continue_interrupted_does_not_trigger_if_frame_is_not_user_frame():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            name: bar flow
            steps:
            - id: "2"
              collect: bar
        """
    )

    stack = DialogueStack(frames=[ChitChatStackFrame(frame_id="some-id")])

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(current_frame, stack, flows)

    # only the original frame should be on the stack
    assert len(stack.frames) == 1


def test_trigger_pattern_completed():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack(frames=[])

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="END", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_completed(current_frame, stack, flows)

    top = stack.top()
    assert top is not None
    assert isinstance(top, CompletedPatternFlowStackFrame)
    assert top.previous_flow_name == "foo flow"


def test_trigger_pattern_completed_does_not_trigger_if_stack_not_empty():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack(frames=[ChitChatStackFrame(frame_id="some-id")])

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="END", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_completed(current_frame, stack, flows)

    # only the original frame should be on the stack
    assert len(stack.frames) == 1


def test_trigger_pattern_completed_does_not_trigger_if_current_frame_is_not_user_frame():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack(frames=[])

    current_frame = ChitChatStackFrame(frame_id="some-other-id")

    flow_executor.trigger_pattern_completed(current_frame, stack, flows)

    # stack should continue to be empty
    assert len(stack.frames) == 0


def test_pattern_ask_collect_information():
    stack = DialogueStack(frames=[])

    collect = "foo"
    utter = "utter_ask_foo"
    rejections = [SlotRejection(if_="1 > 2", utter="42")]

    flow_executor.trigger_pattern_ask_collect_information(
        collect, stack, rejections, utter
    )

    top = stack.top()
    assert top is not None
    assert isinstance(top, CollectInformationPatternFlowStackFrame)
    assert top.collect == collect
    assert top.utter == utter
    assert top.rejections == rejections


def test_reset_scoped_slots():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - id: "1"
              collect: foo
              reset_after_flow_ends: true
            - id: "2"
              collect: bar
              reset_after_flow_ends: false
            - id: "3"
              collect: baz
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            SlotSet("foo", "foo"),
            SlotSet("bar", "bar"),
            SlotSet("baz", "baz"),
        ],
    )
    events = flow_executor.reset_scoped_slots(current_flow, tracker)
    assert events == [SlotSet("foo", None), SlotSet("baz", None)]


def test_reset_scoped_slots_uses_initial_value():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - collect: foo
              reset_after_flow_ends: true
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events(
        "test",
        [SlotSet("foo", "foo")],
        slots=[TextSlot("foo", mappings={}, initial_value="42")],
    )
    events = flow_executor.reset_scoped_slots(current_flow, tracker)
    assert events == [SlotSet("foo", "42")]


def test_reset_scoped_slots_resets_set_slots():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - set_slots: 
              - foo: bar
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "foo")])
    events = flow_executor.reset_scoped_slots(current_flow, tracker)
    assert events == [SlotSet("foo", None)]


def test_reset_scoped_slots_does_not_reset_set_slots_if_collect_forbids_it():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            steps:
            - collect: foo
              reset_after_flow_ends: false
            - set_slots: 
              - foo: bar
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "foo")])
    events = flow_executor.reset_scoped_slots(current_flow, tracker)
    assert events == []


def test_run_step():
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
    flow = user_flow_frame.flow(all_flows)

    result = flow_executor.run_step(step, flow, stack, tracker, domain, flows)


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
        tracker.update_with_events(action_prediction.events or [])
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
