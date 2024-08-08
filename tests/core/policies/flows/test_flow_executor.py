import uuid
from typing import List, Optional, Tuple

import pytest
import structlog

from rasa.core.policies.flows import flow_executor
from rasa.core.policies.flows.flow_exceptions import (
    FlowCircuitBreakerTrippedException,
    NoNextStepInFlowException,
)
from rasa.core.policies.flows.flow_executor import (
    validate_custom_slot_mappings,
    validate_collect_step,
)
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    PauseFlowReturnPrediction,
)
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.completed import (
    CompletedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.chit_chat_frame import ChitChatStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    BaseFlowStackFrame,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.search_frame import SearchStackFrame
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    FlowCompleted,
    FlowResumed,
    FlowStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.flow import (
    END_STEP,
    ContinueFlowStep,
    EndFlowStep,
)
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps import SetSlotsFlowStep
from rasa.shared.core.flows.steps.collect import (
    CollectInformationFlowStep,
    SlotRejection,
)
from rasa.shared.core.flows.yaml_flows_io import (
    flows_from_str,
    flows_from_str_including_defaults,
)
from rasa.shared.core.slots import FloatSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.conftest import update_tracker_with_path_through_flow
from tests.utilities import filter_logs


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
    stack = DialogueStack.empty()
    tracker = DialogueStateTracker.from_events("test", [])
    assert flow_executor.is_condition_satisfied(predicate, stack, tracker)


def test_evaluate_simple_predicate_failing():
    predicate = "2 < 1"
    stack = DialogueStack.empty()
    tracker = DialogueStateTracker.from_events("test", [])
    assert not flow_executor.is_condition_satisfied(predicate, stack, tracker)


def test_invalid_predicate():
    predicate = "2 >!= 1"
    stack = DialogueStack.empty()
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

    satisfied_tracker = DialogueStateTracker.from_events("test", [])
    satisfied_tracker.update_stack(stack)
    assert flow_executor.is_condition_satisfied(
        predicate,
        stack.current_context(),
        satisfied_tracker,
    )


def test_evaluate_predicate_with_slots():
    predicate = "'foo' = slots.my_slot"

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
        next=FlowStepLinks(links=[]),
    )
    assert not flow_executor.is_step_end_of_flow(step)


def test_is_step_end_of_flow_is_true_for_end():
    step = EndFlowStep()
    assert flow_executor.is_step_end_of_flow(step)


def test_is_step_end_of_flow_is_true_for_step_continuing_at_end():
    step = ContinueFlowStep(target_step_id=END_STEP)
    assert flow_executor.is_step_end_of_flow(step)


def test_select_next_step_static_link():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
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
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
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
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
              next:
              - if: slots.foo is 'foobar'
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
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "foobar")])
    tracker.update_stack(stack)
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
            description: flow my_flow
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
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "bazbaz")])
    tracker.update_stack(stack)
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
            description: flow my_flow
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
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "bazbaz")])
    tracker.update_stack(stack)
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
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == END_STEP
    )


def test_select_handles_no_next():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    step = user_flow_frame.step(all_flows)

    # we need to manually create this case as the YAML parser doesn't allow
    # for empty nexts. so actually, we shouldn't even get into this situation
    # but still good to make sure that the function handles it
    step.next = FlowStepLinks(links=[])

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        is None
    )


def test_select_handles_current_node_being_END():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id=END_STEP, frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
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
            description: flow my_flow
            steps:
            - id: link_to_foo
              link: foo

          foo:
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="link_to_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    step = user_flow_frame.step(all_flows)

    assert (
        flow_executor.select_next_step_id(step, stack.current_context(), tracker)
        == END_STEP
    )


def test_advance_top_flow_on_stack_handles_empty_stack():
    stack = DialogueStack.empty()
    flow_executor.update_top_flow_step_id("foo", stack)
    assert stack == DialogueStack.empty()


def test_advance_top_flow_on_stack_handles_non_user_flow_stack():
    search_frame = SearchStackFrame(frame_id="some-frame-id")
    stack = DialogueStack(frames=[search_frame])
    flow_executor.update_top_flow_step_id("foo", stack)
    assert stack == DialogueStack(frames=[search_frame])


def test_advance_top_flow_on_stack_advances_user_flow():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    flow_executor.update_top_flow_step_id("bar", stack)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.step_id == "bar"


def test_trigger_pattern_continue_interrupted_adds_stackframe():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            description: flow bar
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
            description: flow foo
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            description: flow bar
            name: bar flow
            steps:
            - id: "2"
              collect: bar
        """
    )

    user_flow_stack_frame = UserFlowStackFrame(
        flow_id="bar_flow", step_id="2", frame_id="some-id"
    )
    stack = DialogueStack(frames=[user_flow_stack_frame])

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(current_frame, stack, flows)

    # only the original frame should be on the stack
    assert len(stack.frames) == 1
    assert stack.frames[0] == user_flow_stack_frame


def test_trigger_pattern_continue_interrupted_does_not_trigger_if_finished():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            description: flow bar
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
                step_id=END_STEP,
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


def test_trigger_pattern_continue_interrupted_does_not_trigger_if_not_user_frame():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              collect: foo
          bar_flow:
            description: flow bar
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


def test_trigger_pattern_completed_on_user_flow_frame():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            name: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack.empty()

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id=END_STEP, frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_completed(current_frame, stack, flows)

    top = stack.top()
    assert top is not None
    assert isinstance(top, CompletedPatternFlowStackFrame)
    assert top.previous_flow_name == "foo flow"


def test_trigger_pattern_completed_on_search_frame():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack.empty()

    current_frame = SearchPatternFlowStackFrame(
        flow_id="foo_flow", step_id=END_STEP, frame_id="some-other-id"
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
            description: flow foo
            name: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack(frames=[ChitChatStackFrame(frame_id="some-id")])

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id=END_STEP, frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_completed(current_frame, stack, flows)

    # only the original frame should be on the stack
    assert len(stack.frames) == 1


def test_trigger_pattern_completed_does_not_trigger_if_not_user_frame():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            name: foo flow
            steps:
            - id: "1"
              collect: foo
        """
    )

    stack = DialogueStack.empty()

    current_frame = ChitChatStackFrame(frame_id="some-other-id")

    flow_executor.trigger_pattern_completed(current_frame, stack, flows)

    # stack should continue to be empty
    assert len(stack.frames) == 0


def test_pattern_ask_collect_information():
    stack = DialogueStack.empty()

    collect = "foo"
    utter = "utter_ask_foo"
    collect_action = "action_ask_foo"
    rejections = [SlotRejection(if_="1 > 2", utter="42")]

    flow_executor.trigger_pattern_ask_collect_information(
        collect, stack, rejections, utter, collect_action
    )

    top = stack.top()
    assert top is not None
    assert isinstance(top, CollectInformationPatternFlowStackFrame)
    assert top.collect == collect
    assert top.utter == utter
    assert top.rejections == rejections
    assert top.collect_action == collect_action


def test_reset_scoped_slots():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
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
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1", "2", "3"])
    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    assert events == [SlotSet("foo", None), SlotSet("baz", None)]


def test_reset_scoped_slots_uses_initial_value():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            name: foo flow
            steps:
            - id: "1"
              collect: foo
              reset_after_flow_ends: true
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events(
        "test",
        [SlotSet("foo", "foo")],
        slots=[TextSlot("foo", mappings={}, initial_value="42")],
    )
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1"])
    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    assert events == [SlotSet("foo", "42")]


def test_reset_scoped_slots_resets_set_slots():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            name: foo flow
            steps:
            - id: "1"
              set_slots:
              - foo: bar
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "foo")])
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1"])
    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    assert events == [SlotSet("foo", None)]


def test_reset_scoped_slots_does_not_reset_set_slots_if_collect_forbids_it():
    # a flow with three different collect steps, one for each configuration
    # value (true, false and unset)
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            name: foo flow
            steps:
            - id: "1"
              collect: foo
              reset_after_flow_ends: false
            - id: "2"
              set_slots:
              - foo: bar
        """
    )
    current_flow = flows.flow_by_id("foo_flow")
    tracker = DialogueStateTracker.from_events("test", [SlotSet("foo", "foo")])
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1", "2"])
    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    assert events == []


def test_run_step_collect():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("collect_foo")

    available_actions = ["utter_ask_foo"]

    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [FlowStarted(flow_id="my_flow")]
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[0], UserFlowStackFrame)
    assert isinstance(stack.frames[1], CollectInformationPatternFlowStackFrame)


@pytest.mark.parametrize("ask_before_filling", [True, False])
def test_run_step_collect_with_ask_before_filling(ask_before_filling: bool):
    flows = flows_from_str(
        f"""
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
              ask_before_filling: {ask_before_filling}
        """
    )
    slots = [TextSlot("foo", [], initial_value="some value")]
    actions = ["utter_ask_foo"]

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_foo", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [], slots=slots)
    tracker.update_stack(stack)
    step = user_flow_frame.step(flows)
    flow = user_flow_frame.flow(flows)

    result = flow_executor.run_step(step, flow, stack, tracker, actions, flows)
    expected_events = [FlowStarted(flow_id="my_flow")]
    if ask_before_filling:
        expected_events.append(SlotSet("foo", None))

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == expected_events
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[0], UserFlowStackFrame)
    assert isinstance(stack.frames[1], CollectInformationPatternFlowStackFrame)


def test_trigger_pattern_ask_collect_information():
    stack = DialogueStack.empty()
    flow_executor.trigger_pattern_ask_collect_information(
        "collect_foo", stack, [], "utter_ask_foo", "action_ask_foo"
    )

    assert len(stack.frames) == 1
    assert isinstance(stack.frames[0], CollectInformationPatternFlowStackFrame)
    data = stack.frames[0].as_dict()
    assert data["collect"] == "collect_foo"
    assert data["utter"] == "utter_ask_foo"
    assert data["collect_action"] == "action_ask_foo"


def test_run_step_action():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: action
              action: utter_ask_foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("action")

    available_actions = ["utter_ask_foo"]

    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == "utter_ask_foo"


def test_run_step_action_check_warnings():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: fews steps from the pattern collect infomation.
            steps:
            - id: utter
              action: utter_ask_foo
            - id: action
              action: action_ask_foo
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    available_actions = ["utter_ask_foo"]

    # Run the utter step of the collect.
    step = flow.step_by_id("utter")
    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )
    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == "utter_ask_foo"

    # Now run the associated action step of the collect.
    step = flow.step_by_id("action")
    expected_event = "flow.step.run.action.unknown"
    expected_log_level = "warning"
    with structlog.testing.capture_logs() as caplog:
        result = flow_executor.run_step(
            step, flow, stack, tracker, available_actions, flows
        )
        logs = filter_logs(caplog, expected_event, expected_log_level)
        assert len(logs) == 1
        assert isinstance(result, ContinueFlowWithNextStep)


def test_run_step_link():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: link
              link: bar_flow

          bar_flow:
            description: flow bar_flow
            steps:
            - id: action
              action: action_listen
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("link")

    available_actions = []

    # test that my_flow is still on top to be wrapped up and that the linked
    # flow was inserted just below
    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.flow_id == "my_flow"
    linked_flow = stack.frames[0]
    assert isinstance(linked_flow, UserFlowStackFrame)
    assert linked_flow.frame_type == FlowStackFrameType.LINK
    assert linked_flow.flow_id == "bar_flow"


def test_run_step_link_human_handoff():
    flows = flows_from_str_including_defaults(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: link
              link: pattern_human_handoff
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("link")

    available_actions = []

    # test that my_flow is still on top to be wrapped up and that the linked
    # flow was inserted just below
    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.flow_id == "my_flow"
    linked_flow = stack.frames[0]
    assert isinstance(linked_flow, HumanHandoffPatternFlowStackFrame)


def test_run_step_call():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: call
              call: bar_flow

          bar_flow:
            description: flow bar_flow
            steps:
            - id: action
              action: action_listen
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("call")

    available_actions = []

    # test that bar_flow is on top and my_flow is underneath to be continued
    # after bar_flow finished
    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.frame_type == FlowStackFrameType.CALL
    assert top.flow_id == "bar_flow"

    parent_flow = stack.frames[0]
    assert isinstance(parent_flow, UserFlowStackFrame)
    assert parent_flow.frame_type == FlowStackFrameType.REGULAR
    assert parent_flow.flow_id == "my_flow"


def test_run_step_set_slot():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: set_slot
              set_slots:
              - bar: baz
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("set_slot")

    available_actions = []

    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [FlowStarted(flow_id="my_flow"), SlotSet("bar", "baz")]


def test_run_step_end():
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect
              collect: bar
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id(END_STEP)

    available_actions = []

    result = flow_executor.run_step(
        step, flow, stack, tracker, available_actions, flows
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [SlotSet("bar", None)]


def test_executor_does_not_get_tripped_if_an_action_is_predicted_in_loop():
    flow_with_loop = flows_from_str(
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
        evts=[ActionExecuted(action_name="action_listen")],
        domain=domain,
        slots=domain.slots,
    )
    tracker.update_stack(stack)

    available_actions = ["action_listen"]

    selection = flow_executor.advance_flows_until_next_action(
        tracker, available_actions, flow_with_loop
    )
    assert selection.action_name == "action_listen"


def test_executor_trips_internal_circuit_breaker():
    flow_with_loop = flows_from_str(
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

    domain = Domain.empty()

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

    available_actions = []

    with pytest.raises(FlowCircuitBreakerTrippedException):
        flow_executor.advance_flows_until_next_action(
            tracker, available_actions, flow_with_loop
        )


def test_executor_raises_no_next_step_in_flow_exception():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              set_slots:
                - foo: bar
              next:
              - if: 1 > 2
                then:
                  - action: utter_something
                    next: END
              - else: "2"
            - id: "2"
              action: utter_hello
        """
    )

    # remove the else branch from the next step so that the expected exception is raised
    next_step = flows.underlying_flows[0].step_by_id("1").next
    next_step.links = [next_step.links[0]]

    domain = Domain.empty()

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

    available_actions = []

    with pytest.raises(NoNextStepInFlowException):
        flow_executor.advance_flows_until_next_action(tracker, available_actions, flows)


def test_advance_flows_empty_stack():
    flows = flows_from_str(
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
    stack = DialogueStack.empty()
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(stack)
    available_actions = []
    prediction = flow_executor.advance_flows(tracker, available_actions, flows)
    assert prediction.action_name is None


def test_advance_flows_selects_next_action():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              collect: foo
            - id: "2"
              action: utter_goodbye
        """
    )
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(stack)
    available_actions = ["utter_goodbye"]
    prediction = flow_executor.advance_flows(tracker, available_actions, flows)
    assert prediction.action_name == "utter_goodbye"

    assert len(prediction.events) == 1
    assert isinstance(prediction.events[0], DialogueStackUpdated)
    tracker.update_with_events(prediction.events)

    assert len(tracker.stack.frames) == 1

    assert isinstance(tracker.stack.frames[0], UserFlowStackFrame)
    assert tracker.stack.frames[0].step_id == "2"
    assert tracker.stack.frames[0].flow_id == "foo_flow"


def _run_flow_until_listen(
    tracker: DialogueStateTracker, domain: Domain, flows: FlowsList
) -> Tuple[List[Optional[str]], List[Event]]:
    # Run the flow until we reach a listen action.
    # Collect and return all events and intermediate actions.
    events = []
    actions = []
    while True:
        action_prediction = flow_executor.advance_flows(
            tracker, domain.action_names_or_texts, flows
        )
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


def test_flow_policy_events_after_flow_starts() -> None:
    flows = flows_from_str(
        """
        flows:
          search_hotels:
            description: flow search_hotels
            steps:
            - id: "1_collect_num_rooms"
              collect: num_rooms
            - collect: start_date
            - collect: end_date
            - action: action_search_hotels
        """
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("search hotels"),
        ],
        slots=[
            FloatSlot("num_rooms", mappings=[]),
            TextSlot("start_date", mappings=[]),
            TextSlot("end_slot", mappings=[]),
        ],
    )

    flow = flows.flow_by_id("search_hotels")
    step = flow.step_by_id("1_collect_num_rooms")
    available_actions = []
    step_result = flow_executor.run_step(
        step=step,
        flow=flow,
        stack=tracker.stack,
        tracker=tracker,
        available_actions=available_actions,
        flows=flows,
    )
    assert step_result is not None
    assert step_result.events == [FlowStarted("search_hotels")]


def test_flow_policy_events_after_flow_ends() -> None:
    flows = flows_from_str(
        """
        flows:
          search_hotels:
            description: flow search_hotels
            steps:
            - id: "1_collect_num_rooms"
              collect: num_rooms
            - collect: start_date
            - collect: end_date
            - id: "2_action_search_hotels"
              action: action_search_hotels
          pattern_completed:
            description: flow pattern_completed
            steps:
            - action: utter_completed
        """
    )
    stack = DialogueStack.from_dict(
        [
            {
                "flow_id": "search_hotels",
                "frame_id": "OGE1U359",
                "frame_type": "regular",
                "step_id": "2_action_search_hotels",
                "type": "flow",
            }
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("search hotels"),
            BotUttered("How many rooms?"),
            UserUttered("5"),
            BotUttered("When do you want to check in?"),
            UserUttered("tomorrow"),
            BotUttered("When do you want to check out?"),
            UserUttered("next week"),
            BotUttered("Ok, searching for hotels"),
        ],
        slots=[
            FloatSlot("num_rooms", mappings=[]),
            TextSlot("start_date", mappings=[]),
            TextSlot("end_slot", mappings=[]),
        ],
    )
    tracker.update_stack(stack)

    available_actions = []
    result = flow_executor.advance_flows_until_next_action(
        tracker=tracker,
        available_actions=available_actions,
        flows=flows,
    )
    assert result is not None
    assert (
        FlowCompleted(flow_id="search_hotels", step_id="2_action_search_hotels")
        in result.events
    )


def test_flow_policy_events_after_interruption() -> None:
    flows = flows_from_str(
        """
        flows:
          search_hotels:
            description: flow search_hotels
            steps:
            - id: "1_collect_num_rooms"
              collect: num_rooms
            - collect: start_date
            - collect: end_date
            - action: action_search_hotels
          check_balance:
            description: flow check_balance
            steps:
            - id: "1_check_balance"
              action: action_check_balance
            - id: "2_utter_current_balance"
              action: utter_current_balance
          pattern_continue_interrupted:
            description: flow pattern_continue_interrupted
            steps:
            - action: utter_continue_interrupted
          pattern_collect_information:
            description: flow pattern_collect_information
            steps:
            - action: utter_collect_information
            - id: "listen"
              action: action_listen
          pattern_completed:
            description: flow pattern_completed
            steps:
            - action: utter_how_else_can_i_help
        """
    )
    stack = DialogueStack.from_dict(
        [
            {
                "flow_id": "search_hotels",
                "frame_id": "OGE1U359",
                "frame_type": "regular",
                "step_id": "1_collect_num_rooms",
                "type": "flow",
            },
            {
                "collect": "num_rooms",
                "flow_id": "pattern_collect_information",
                "frame_id": "39LEDJUN",
                "rejections": [],
                "step_id": "listen",
                "type": "pattern_collect_information",
                "utter": "utter_ask_num_rooms",
                "collect_action": "action_ask_num_rooms",
            },
            {
                "flow_id": "check_balance",
                "frame_id": "6ZV8O9T3",
                "frame_type": "interrupt",
                "step_id": "2_utter_current_balance",
                "type": "flow",
            },
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("search hotels"),
            BotUttered("How many rooms?"),
            UserUttered("Check how much money I have"),
            BotUttered("You have 1000 dollars"),
            BotUttered("How many rooms?"),
        ],
        slots=[
            FloatSlot("num_rooms", mappings=[]),
            TextSlot("start_date", mappings=[]),
            TextSlot("end_slot", mappings=[]),
        ],
    )
    tracker.update_stack(stack)

    available_actions = [
        "utter_ask_num_rooms",
        "utter_ask_start_date",
        "utter_ask_end_date",
        "action_check_balance",
        "utter_current_balance",
        "action_search_hotels",
        "utter_continue_interrupted",
        "action_listen",
        "utter_how_else_can_i_help",
        "utter_collect_information",
    ]
    result = flow_executor.advance_flows_until_next_action(
        tracker=tracker,
        available_actions=available_actions,
        flows=flows,
    )
    assert result is not None
    assert result.events[2] == FlowCompleted(
        flow_id="check_balance", step_id="2_utter_current_balance"
    )
    assert result.events[3] == FlowResumed(
        flow_id="search_hotels", step_id="1_collect_num_rooms"
    )


@pytest.mark.parametrize(
    "predicate, expected", [("slots.bar > 10", True), ("slots.bar <= 10", False)]
)
def test_flow_executor_is_condition_satisfied_with_slots_namespace(
    predicate: str,
    expected: bool,
) -> None:
    test_domain = Domain.from_yaml(
        """
        slots:
            bar:
              type: float
              initial_value: 0.0
        """
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[SlotSet("bar", 100)],
        slots=test_domain.slots,
    )

    context = {}
    result = flow_executor.is_condition_satisfied(predicate, context, tracker)

    assert result is expected


@pytest.mark.parametrize(
    "predicate, expected",
    [
        ("slots.payment_type == 'Direct Debit'", True),
        ("slots.user_type == 'Premium'", True),
        ("slots.payment_type == 'DIRECT DEBIT'", False),
        ("slots.user_type == 'premium'", False),
    ],
)
def test_flow_executor_is_condition_satisfied_with_categorical_slots(
    predicate: str,
    expected: bool,
) -> None:
    test_domain = Domain.from_yaml(
        """
        slots:
            payment_type:
              type: categorical
              values:
                - Direct Debit
                - Credit card
                - international transfer
            user_type:
              type: categorical
              values:
                - Premium
                - Standard
        """
    )

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[SlotSet("payment_type", "DIRECT DEBIT"), SlotSet("user_type", "premium")],
        slots=test_domain.slots,
    )

    context = {}
    assert flow_executor.is_condition_satisfied(predicate, context, tracker) == expected


def test_flow_executor_validate_custom_slot_mappings_valid() -> None:
    domain = Domain.from_yaml(
        """
    slots:
        loyalty_points:
            type: float
            mappings:
                - type: custom
    actions:
        - action_ask_loyalty_points
    """
    )
    step = CollectInformationFlowStep.from_json({"collect": "loyalty_points"})
    stack = DialogueStack(frames=[UserFlowStackFrame(flow_id="my_flow", step_id="1")])
    tracker = DialogueStateTracker.from_events("test", [], slots=domain.slots)
    tracker.update_stack(stack)

    is_valid = validate_custom_slot_mappings(
        step, stack, tracker, domain.action_names_or_texts
    )

    assert is_valid
    assert stack.current_context().get("flow_id") == "my_flow"


def test_flow_executor_validate_custom_slot_mappings_invalid() -> None:
    domain = Domain.from_yaml(
        """
    slots:
        loyalty_points:
            type: float
            mappings:
                - type: custom
    responses:
        utter_ask_loyalty_points:
            - text: "Let's proceed checking how many loyalty points you have."
    """
    )
    step = CollectInformationFlowStep.from_json({"collect": "loyalty_points"})
    stack = DialogueStack(frames=[UserFlowStackFrame(flow_id="my_flow", step_id="1")])
    tracker = DialogueStateTracker.from_events("test", [], slots=domain.slots)
    tracker.update_stack(stack)

    is_valid = validate_custom_slot_mappings(
        step, stack, tracker, domain.action_names_or_texts
    )

    assert not is_valid
    assert stack.current_context().get("flow_id") == "pattern_internal_error"

    bottom_frame = stack.frames[0]
    assert isinstance(bottom_frame, UserFlowStackFrame)
    assert bottom_frame.flow_id == "my_flow"
    assert bottom_frame.step_id == "1"

    next_frame = stack.frames[1]
    assert isinstance(next_frame, BaseFlowStackFrame)
    assert next_frame.flow_id == "pattern_cancel_flow"


def test_flow_executor_validate_collect_step_invalid() -> None:
    test_domain = Domain.from_yaml(
        """
    slots:
        loyalty_points:
            type: float
            mappings:
                - type: from_llm
    """
    )
    step = CollectInformationFlowStep.from_json({"collect": "loyalty_points"})
    stack = DialogueStack(frames=[UserFlowStackFrame(flow_id="my_flow", step_id="1")])
    tracker = DialogueStateTracker.from_events("test", [], slots=test_domain.slots)
    tracker.update_stack(stack)

    is_step_valid = validate_collect_step(
        step, stack, test_domain.action_names_or_texts, tracker.slots
    )

    assert not is_step_valid
    assert stack.current_context().get("flow_id") == "pattern_internal_error"

    bottom_frame = stack.frames[0]
    assert isinstance(bottom_frame, UserFlowStackFrame)
    assert bottom_frame.flow_id == "my_flow"
    assert bottom_frame.step_id == "1"

    next_frame = stack.frames[1]
    assert isinstance(next_frame, BaseFlowStackFrame)
    assert next_frame.flow_id == "pattern_cancel_flow"


def test_flow_executor_validate_collect_step_with_initial_value_defined() -> None:
    test_domain = Domain.from_yaml(
        """
    slots:
        loyalty_points:
            type: float
            initial_value: 0.0
            mappings:
                - type: from_llm
    """
    )
    step = CollectInformationFlowStep.from_json({"collect": "loyalty_points"})
    stack = DialogueStack(frames=[UserFlowStackFrame(flow_id="my_flow", step_id="1")])
    tracker = DialogueStateTracker.from_events("test", [], slots=test_domain.slots)
    tracker.update_stack(stack)

    is_valid = validate_collect_step(
        step, stack, test_domain.action_names_or_texts, tracker.slots
    )

    assert is_valid
    assert stack.current_context().get("flow_id") == "my_flow"
