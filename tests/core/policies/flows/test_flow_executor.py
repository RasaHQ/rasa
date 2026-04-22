import uuid
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import structlog
from pytest import MonkeyPatch

from rasa.core.config.available_endpoints import (
    InteractionHandlingConfig,
)
from rasa.core.config.configuration import Configuration
from rasa.core.config.credentials import CredentialsConfig
from rasa.core.policies.flows import flow_executor
from rasa.core.policies.flows.flow_exceptions import (
    FlowCircuitBreakerTrippedException,
    NoNextStepInFlowException,
)
from rasa.core.policies.flows.flow_executor import (
    _restore_suspended_agent_frame_if_any,
    get_next_step_id_after_agent,
    select_next_step,
    select_next_step_id,
    validate_collect_step,
)
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    PauseFlowReturnPrediction,
)
from rasa.dialogue_understanding.patterns.cannot_handle import (
    CannotHandlePatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.clarify import ClarifyPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.completed import (
    CompletedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.customer_satisfaction import (
    CustomerSatisfactionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.chit_chat_frame import ChitChatStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    BaseFlowStackFrame,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.search_frame import SearchStackFrame
from rasa.engine.language import Language
from rasa.shared.constants import (
    RASA_PATTERN_CANNOT_HANDLE_NO_RELEVANT_ANSWER,
)
from rasa.shared.core.constants import (
    ACTION_SESSION_START_NAME,
    GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE,
    SILENCE_TIMEOUT_CHANNEL_KEY,
    SILENCE_TIMEOUT_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    DialogueStackUpdated,
    Event,
    FlowCompleted,
    FlowStarted,
    SessionEnded,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList, FlowStep
from rasa.shared.core.flows.flow import (
    END_STEP,
    ContinueFlowStep,
    EndFlowStep,
)
from rasa.shared.core.flows.flow_step_links import FlowStepLinks, StaticFlowStepLink
from rasa.shared.core.flows.steps import SetSlotsFlowStep
from rasa.shared.core.flows.steps.collect import (
    CollectInformationFlowStep,
)
from rasa.shared.core.flows.steps.constants import START_STEP
from rasa.shared.core.flows.yaml_flows_io import YAMLFlowsReader
from rasa.shared.core.slots import (
    BooleanSlot,
    FloatSlot,
    SlotRejection,
    StrictCategoricalSlot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.conftest import update_tracker_with_path_through_flow
from tests.utilities import (
    filter_logs,
    flows_from_str,
    flows_from_str_including_defaults,
)


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
        flow_id="my_flow",
    )
    assert not flow_executor.is_step_end_of_flow(step)


def test_is_step_end_of_flow_is_true_for_end():
    step = EndFlowStep("my_flow")
    assert flow_executor.is_step_end_of_flow(step)


def test_is_step_end_of_flow_is_true_for_step_continuing_at_end():
    step = ContinueFlowStep(flow_id="my_flow", target_step_id=END_STEP)
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


def test_select_next_step_id_returns_current_id_when_waiting_agent_stack_frame_on_top():
    """Top AgentStackFrame in WAITING_FOR_INPUT loops back to the current step."""
    agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="agent-1",
        flow_id="flow-1",
    )
    step = FlowStep(
        custom_id="my_step_id",
        idx=0,
        description=None,
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="my_flow",
    )
    stack = DialogueStack(frames=[agent_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    result = select_next_step_id(step, stack.current_context(), tracker)
    assert result == "my_step_id"


def test_select_next_step_id_follows_links_when_interrupted_agent_stack_frame_on_top():
    """ENG-2713: Interrupted agent on stack must not force loop-back; follow links."""
    agent_stack_frame = AgentStackFrame(
        frame_id="agent-frame-id",
        state=AgentState.INTERRUPTED,
        agent_id="agent-1",
        flow_id="flow-1",
    )
    step = FlowStep(
        custom_id="my_step_id",
        idx=0,
        description=None,
        metadata={},
        next=FlowStepLinks(
            links=[StaticFlowStepLink(target_step_id="next_after_call")]
        ),
        flow_id="my_flow",
    )
    stack = DialogueStack(frames=[agent_stack_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    result = select_next_step_id(step, stack.current_context(), tracker)
    assert result == "next_after_call"


def test_select_next_step_id_ignore_agent_on_stack_returns_next_step():
    """ENG-2669: With ignore_agent_on_stack=True, returns next step not current."""
    agent_frame = AgentStackFrame(
        flow_id="my_flow",
        step_id="call_agent_a",
        agent_id="agent_a",
        state=AgentState.INTERRUPTED,
        frame_id="agent-frame",
    )
    stack = DialogueStack(frames=[agent_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    step = FlowStep(
        custom_id="call_agent_a",
        idx=0,
        description=None,
        metadata={},
        next=FlowStepLinks(links=[StaticFlowStepLink(target_step_id="step_between")]),
        flow_id="my_flow",
    )
    result = select_next_step_id(
        step, stack.current_context(), tracker, ignore_agent_on_stack=True
    )
    assert result == "step_between"


def test_get_next_step_id_after_agent_returns_next_step():
    """ENG-2669: Returns the step id after the agent in the flow."""
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test flow
            steps:
            - id: call_agent_a
              call: agent_a
            - id: step_between
              collect: slot_x
            - id: call_agent_b
              call: agent_b
          agent_a:
            description: agent a
            steps:
            - id: step_a
              action: action_listen
          agent_b:
            description: agent b
            steps:
            - id: step_b
              action: action_listen
        """
    )
    tracker = DialogueStateTracker.from_events("test", [])
    result = get_next_step_id_after_agent(
        "my_flow", "agent_a", all_flows, tracker, step_id="call_agent_a"
    )
    assert result == "step_between"
    # agent_b is last in flow; next step is END
    result_after_b = get_next_step_id_after_agent(
        "my_flow", "agent_b", all_flows, tracker, step_id="call_agent_b"
    )
    assert result_after_b == END_STEP


def test_get_next_step_id_after_agent_matches_step_id_when_agent_called_twice():
    """Same agent called twice in a flow; step_id selects the correct next step."""
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test flow
            steps:
            - id: call_agent_x
              call: agent_x
            - id: step_one
              action: action_listen
            - id: call_agent_x_again
              call: agent_x
            - id: step_two
              action: action_listen
          agent_x:
            description: agent x
            steps:
            - id: step_x
              action: action_listen
        """
    )
    tracker = DialogueStateTracker.from_events("test", [])
    # First call to agent_x -> next is step_one
    result_first = get_next_step_id_after_agent(
        "my_flow", "agent_x", all_flows, tracker, step_id="call_agent_x"
    )
    assert result_first == "step_one"
    # Second call to agent_x -> next is step_two
    result_second = get_next_step_id_after_agent(
        "my_flow", "agent_x", all_flows, tracker, step_id="call_agent_x_again"
    )
    assert result_second == "step_two"


def test_restore_suspended_agent_frame_if_any_pushes_and_removes_from_list():
    """ENG-2669: Restoring a suspended agent frame pushes it and removes from list."""
    agent_b_frame = AgentStackFrame(
        flow_id="my_flow",
        step_id="call_b",
        agent_id="agent_b",
        state=AgentState.INTERRUPTED,
        frame_id="agent-b-frame",
    )
    user_flow = UserFlowStackFrame(
        flow_id="my_flow",
        step_id="step_between",
        frame_id="user-flow-id",
        suspended_agent_frames=[agent_b_frame.as_dict()],
    )
    stack = DialogueStack(frames=[user_flow])
    _restore_suspended_agent_frame_if_any(stack, "my_flow", "call_b", "agent_b")
    assert len(stack.frames) == 2
    top = stack.top()
    assert isinstance(top, AgentStackFrame)
    assert top.agent_id == "agent_b"
    assert top.state == AgentState.INTERRUPTED
    assert user_flow.suspended_agent_frames == []


def test_restore_suspended_agent_frame_if_any_matches_step_id():
    """Restore only when flow_id, step_id and agent_id match."""
    agent_b_at_call_b = AgentStackFrame(
        flow_id="my_flow",
        step_id="call_b",
        agent_id="agent_b",
        state=AgentState.INTERRUPTED,
        frame_id="agent-b-frame",
    )
    user_flow = UserFlowStackFrame(
        flow_id="my_flow",
        step_id="step_between",
        frame_id="user-flow-id",
        suspended_agent_frames=[agent_b_at_call_b.as_dict()],
    )
    stack = DialogueStack(frames=[user_flow])
    # Different step_id (e.g. second call to agent_b in same flow) -> do not restore
    _restore_suspended_agent_frame_if_any(stack, "my_flow", "call_b_again", "agent_b")
    assert len(stack.frames) == 1
    assert len(user_flow.suspended_agent_frames) == 1
    assert user_flow.suspended_agent_frames[0].get("step_id") == "call_b"


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
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow",
        step_id="1",
        frame_id="some-other-id",
        frame_type=FlowStackFrameType.INTERRUPT,
    )

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

    top = stack.top()
    assert top is not None
    assert isinstance(top, ContinueInterruptedPatternFlowStackFrame)
    assert top.interrupted_flow_names == ["bar flow"]
    assert top.interrupted_flow_ids == ["bar_flow"]


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
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

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
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

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
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    current_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-other-id"
    )

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

    # only the original frame should be on the stack
    assert len(stack.frames) == 1


def test_trigger_pattern_continue_interrupted_triggers_after_search_pattern():
    """Trigger continue_interrupted when current frame is SearchPatternFlowStackFrame.

    This covers the case after a knowledge/EnterpriseSearch answer: the search
    pattern frame is popped and we should offer to continue interrupted flows.
    """
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
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    current_frame = SearchPatternFlowStackFrame()

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

    top = stack.top()
    assert top is not None
    assert isinstance(top, ContinueInterruptedPatternFlowStackFrame)
    assert top.interrupted_flow_names == ["bar flow"]
    assert top.interrupted_flow_ids == ["bar_flow"]


def test_trigger_pattern_continue_interrupted_after_search_no_continuable():
    """Do not trigger continue_interrupted after search pattern if no flows to continue.

    When the current frame is SearchPatternFlowStackFrame but all user flows on
    the stack are at the end (nothing to continue), we should not push the
    pattern.
    """
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
            )
        ]
    )
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    current_frame = SearchPatternFlowStackFrame()

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

    assert len(stack.frames) == 1


def test_trigger_pattern_continue_interrupted_triggers_correctly_with_link_step():
    """Test if pattern_continue_interrupted triggers correctly with link step.

    Conversation being tested (expected behaviour):
    User: Remove contact
    Bot: What's the handle of the user you want to remove?
    User: You didn't understand me correctly
    Bot: <utter_test_b>
    Bot: <utter_test_c> (1)
    Bot: Let's continue with remove a contact. (2)
    Bot: What's the handle of the user you want to remove?

    The primary objective is to verify that the bot correctly invokes
    `pattern_continue_interrupted` after <utter_test_c> (1) in flow_c,
    and then resumes flow_a from the point where it was interrupted.
    """
    flows = flows_from_str(
        """
        flows:
          flow_a:
            name: remove a contact
            description: remove a contact from your contact list
            steps:
              - collect: "remove_contact_handle"
                description: "a contact handle starting with @"
              - collect: "remove_contact_confirmation"
                ask_before_filling: true
          flow_b:
            description: dummy flow b
            steps:
              - action: utter_test_b
              - link: flow_c
          flow_c:
            description: dummy link flow c
            if: False
            steps:
              - action: utter_test_c
        """
    )

    frame1 = UserFlowStackFrame(
        flow_id="flow_a",
        frame_type=FlowStackFrameType.REGULAR,
        step_id="flow_a_0_collect_remove_contact_handle",
        frame_id="id0",
    )
    frame2 = CollectInformationPatternFlowStackFrame(
        flow_id="pattern_collect_information", step_id="4_action_listen", frame_id="id1"
    )
    link_frame = UserFlowStackFrame(
        flow_id="flow_c",
        frame_type=FlowStackFrameType.LINK,
        step_id="START",
        frame_id="id2",
    )
    stack = DialogueStack(frames=[frame1, frame2, link_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    current_frame = UserFlowStackFrame(
        flow_id="flow_b",
        frame_type=FlowStackFrameType.INTERRUPT,
        step_id="END",
        frame_id="some-id",
    )
    continue_interrupted = ContinueInterruptedPatternFlowStackFrame(
        flow_id="pattern_continue_interrupted",
        step_id="START",
        frame_id="some-id",
        interrupted_flow_names=["remove a contact"],
        interrupted_flow_ids=["flow_a"],
        interrupted_flow_options="remove a contact",
        multiple_flows_interrupted=False,
    )

    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

    assert len(stack.frames) == 4
    assert stack.frames[-1] == link_frame
    assert stack.frames[-2] == continue_interrupted


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


def test_trigger_pattern_completed_does_not_trigger_when_explicitly_prohibited():
    """Tests that `trigger_pattern_completed` does not trigger when the flow has `run_pattern_completed: False`."""  # noqa: E501
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            run_pattern_completed: False
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
            SlotSet(
                "foo",
                "foo",
            ),
            SlotSet("bar", "bar"),
            SlotSet("baz", "baz"),
        ],
    )
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1", "2", "3"])
    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    assert events == [SlotSet("foo", None), SlotSet("baz", None)]
    assert events[0].metadata == {"reset": True}
    assert events[1].metadata == {"reset": True}


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
    assert events[0].metadata == {"reset": True}


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
    assert events[0].metadata == {"reset": True}


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


def test_reset_scoped_slots_with_persisted_slots_set():
    flows = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            name: foo flow
            persisted_slots:
            - foo
            - bar
            steps:
            - id: "1"
              collect: foo
            - id: "2"
              collect: bar
            - id: "3"
              collect: baz
            - id: "4"
              set_slots:
              - foo: foo2
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
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1", "2", "3", "4"])
    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    assert events == [SlotSet("baz", None)]
    assert events[0].metadata == {"reset": True}


def test_reset_scoped_slots_with_persisted_slots_set_in_called_flow():
    flows = flows_from_str(
        """
        flows:
          parent_flow:
            description: parent flow
            name: parent flow
            steps:
            - id: "1"
              call: child_flow

          child_flow:
            description: child flow
            name: child flow
            persisted_slots:
            - foo
            - bar
            steps:
            - id: "1"
              collect: foo
            - id: "2"
              collect: bar
            - id: "3"
              collect: baz
            - id: "4"
              set_slots:
              - foo: foo2
        """
    )
    current_flow = flows.flow_by_id("parent_flow")
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            SlotSet("foo", "foo"),
            SlotSet("bar", "bar"),
            SlotSet("baz", "baz"),
        ],
    )
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "parent_flow",
                    "step_id": "first_step",
                    "frame_id": "some-frame-id",
                },
                {
                    "type": "flow",
                    "frame_type": "call",
                    "flow_id": "child_flow",
                    "step_id": "second_step",
                    "frame_id": "some-other-frame-id",
                },
            ],
        )
    )
    # update tracker with the steps taken in the called flow
    update_tracker_with_path_through_flow(tracker, "foo_flow", ["1", "2", "3", "4"])

    events = flow_executor.reset_scoped_slots(
        tracker.stack.top(), current_flow, tracker
    )
    # we expect only the baz slot to be reset since it wasn't defined to be persisted
    assert events == [SlotSet("baz", None)]
    assert events[0].metadata == {"reset": True}


@pytest.mark.asyncio
async def test_run_step_collect():
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

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert FlowStarted(flow_id="my_flow") in result.events
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[0], UserFlowStackFrame)
    assert isinstance(stack.frames[1], CollectInformationPatternFlowStackFrame)


@pytest.mark.parametrize(
    "ask_before_filling, expected_events",
    [
        (True, [FlowStarted(flow_id="my_flow"), SlotSet("foo", None)]),
        (False, [FlowStarted(flow_id="my_flow")]),
    ],
)
@pytest.mark.asyncio
async def test_run_step_collect_with_ask_before_filling(
    ask_before_filling: bool, expected_events: list[Event]
):
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

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        actions,
        flows,
        previous_step_id=START_STEP,
        slots=slots,
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    # Check that the expected events are in the result
    assert set(expected_events) <= set(result.events)
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


@pytest.mark.asyncio
async def test_run_step_action():
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

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == "utter_ask_foo"


@pytest.mark.asyncio
async def test_run_step_action_check_warnings():
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
    utter_step = flow.step_by_id("utter")
    result = await flow_executor.run_step(
        utter_step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )
    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == "utter_ask_foo"

    # Now run the associated action step of the collect.
    step = flow.step_by_id("action")
    expected_event = "flow.step.run.action.unknown"
    expected_log_level = "warning"
    with structlog.testing.capture_logs() as caplog:
        result = await flow_executor.run_step(
            step,
            flow,
            stack,
            tracker,
            available_actions,
            flows,
            previous_step_id=utter_step.id,
            slots=[],
        )
        logs = filter_logs(caplog, expected_event, expected_log_level)
        assert len(logs) == 1
        assert isinstance(result, ContinueFlowWithNextStep)


@pytest.mark.asyncio
async def test_run_step_link():
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
    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.flow_id == "my_flow"
    linked_flow = stack.frames[0]
    assert isinstance(linked_flow, UserFlowStackFrame)
    assert linked_flow.frame_type == FlowStackFrameType.LINK
    assert linked_flow.flow_id == "bar_flow"


@pytest.mark.asyncio
async def test_run_step_link_human_handoff():
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
    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.flow_id == "my_flow"
    linked_flow = stack.frames[0]
    assert isinstance(linked_flow, HumanHandoffPatternFlowStackFrame)


@pytest.mark.asyncio
async def test_run_step_call(mock_available_agents):
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

    with patch(
        "rasa.core.policies.flows.flow_executor.run_agent", new_callable=AsyncMock
    ) as mock_run_agent:
        # test that bar_flow is on top and my_flow is underneath to be continued
        # after bar_flow finished
        result = await flow_executor.run_step(
            step,
            flow,
            stack,
            tracker,
            available_actions,
            flows,
            previous_step_id=START_STEP,
            slots=[],
        )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events[0] == FlowStarted("my_flow")
    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.frame_type == FlowStackFrameType.CALL
    assert top.flow_id == "bar_flow"

    parent_flow = stack.frames[0]
    assert isinstance(parent_flow, UserFlowStackFrame)
    assert parent_flow.frame_type == FlowStackFrameType.REGULAR
    assert parent_flow.flow_id == "my_flow"
    # Ensure that the agent was used
    mock_run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_step_call_when_agent_exists(
    monkeypatch: MonkeyPatch, mock_available_agents
):
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

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("my-call-step")

    with patch(
        "rasa.core.policies.flows.flow_executor.run_agent", new_callable=AsyncMock
    ) as mock_run_agent:
        # WHEN
        await flow_executor.run_step(
            step,
            flow,
            stack,
            tracker,
            [],
            flows,
            previous_step_id=START_STEP,
            slots=[],
        )

        # THEN
        mock_run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_step_set_slot():
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

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE),
        SlotSet("bar", "baz"),
    ]


@pytest.mark.asyncio
async def test_run_step_end():
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

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )
    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [
        FlowStarted("my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE),
        SlotSet("bar", None),
    ]


@pytest.mark.asyncio
async def test_executor_does_not_get_tripped_if_an_action_is_predicted_in_loop():
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

    selection = await flow_executor.advance_flows_until_next_action(
        tracker, available_actions, flow_with_loop, slots=[]
    )
    assert selection.action_name == "action_listen"


@pytest.mark.asyncio
async def test_executor_trips_internal_circuit_breaker():
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
        await flow_executor.advance_flows_until_next_action(
            tracker, available_actions, flow_with_loop, slots=[]
        )


@pytest.mark.asyncio
async def test_executor_raises_no_next_step_in_flow_exception():
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
        await flow_executor.advance_flows_until_next_action(
            tracker, available_actions, flows, slots=[]
        )


@pytest.mark.asyncio
async def test_advance_flows_empty_stack():
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
    prediction = await flow_executor.advance_flows(
        tracker, available_actions, flows, slots=[]
    )
    assert prediction.action_name is None


@pytest.mark.asyncio
async def test_advance_flows_selects_next_action():
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
    prediction = await flow_executor.advance_flows(
        tracker, available_actions, flows, slots=[]
    )
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


@pytest.mark.asyncio
async def test_flow_policy_events_after_flow_starts() -> None:
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
    step_result = await flow_executor.run_step(
        step=step,
        flow=flow,
        stack=tracker.stack,
        tracker=tracker,
        available_actions=available_actions,
        flows=flows,
        previous_step_id=START_STEP,
        slots=[],
    )
    assert step_result is not None
    assert step_result.events == [FlowStarted("search_hotels")]


@pytest.mark.asyncio
async def test_flow_policy_events_after_flow_ends() -> None:
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
    result = await flow_executor.advance_flows_until_next_action(
        tracker=tracker,
        available_actions=available_actions,
        flows=flows,
        slots=[],
    )
    assert result is not None
    assert (
        FlowCompleted(flow_id="search_hotels", step_id="2_action_search_hotels")
        in result.events
    )


@pytest.mark.asyncio
async def test_flow_policy_events_after_interruption() -> None:
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
    result = await flow_executor.advance_flows_until_next_action(
        tracker=tracker,
        available_actions=available_actions,
        flows=flows,
        slots=[],
    )
    assert result is not None
    assert result.events[2] == FlowCompleted(
        flow_id="check_balance", step_id="2_utter_current_balance"
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
    step = CollectInformationFlowStep.from_json(
        "my_flow", {"collect": "loyalty_points"}
    )
    stack = DialogueStack(frames=[UserFlowStackFrame(flow_id="my_flow", step_id="1")])
    tracker = DialogueStateTracker.from_events("test", [], slots=test_domain.slots)
    tracker.update_stack(stack)

    is_step_valid = validate_collect_step(
        step,
        stack,
        test_domain.action_names_or_texts,
        tracker.slots,
        flow_name="my flow",
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
    assert next_frame.canceled_name == "my flow"


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
    step = CollectInformationFlowStep.from_json(
        "my_flow", {"collect": "loyalty_points"}
    )
    stack = DialogueStack(frames=[UserFlowStackFrame(flow_id="my_flow", step_id="1")])
    tracker = DialogueStateTracker.from_events("test", [], slots=test_domain.slots)
    tracker.update_stack(stack)

    is_valid = validate_collect_step(
        step,
        stack,
        test_domain.action_names_or_texts,
        tracker.slots,
        flow_name="my flow",
    )

    assert is_valid
    assert stack.current_context().get("flow_id") == "my_flow"


@pytest.mark.asyncio
async def test_run_step_adds_metadata_to_flow_started_event():
    flows = flows_from_str(
        """
        flows:
          pattern_clarification:
            description: Conversation repair flow
            name: pattern clarification
            steps:
            - id: start
              action: action_clarify_flows
            - action: utter_clarification_options_rasa
        """
    )
    pattern_clarification_frame = ClarifyPatternFlowStackFrame(
        step_id="start", frame_id="some-frame-id", names=["foo", "bar"]
    )
    stack = DialogueStack(frames=[pattern_clarification_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("pattern_clarification")

    assert flow is not None
    step = flow.step_by_id("start")

    available_actions = ["action_clarify_flows"]

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    expected_events = [
        FlowStarted(flow_id="pattern_clarification", metadata=stack.current_context()),
        SlotSet(SILENCE_TIMEOUT_SLOT, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE),
    ]
    assert result.events == expected_events

    assert expected_events[0].metadata.get("names") == ["foo", "bar"]


@pytest.mark.asyncio
async def test_run_step_does_not_emit_flow_started_event_after_flow_has_started():
    # we reenter the collect_foo step because there is a loop. in this case
    # no flow started event should be emitted - even if we are at the first
    # step of the flow
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
            - id: utter_foo
              action: utter_foo
              next: collect_foo
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

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        actions,
        flows,
        previous_step_id="utter_foo",
        slots=slots,
    )

    assert isinstance(result, ContinueFlowWithNextStep)
    # Check that no FlowStarted event is emitted
    # The only event emitted is the SlotSet for the silence timeout
    assert result.events == [
        SlotSet(SILENCE_TIMEOUT_SLOT, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE)
    ]


async def test_correct_next_step_selected_with_call_step(mock_available_agents) -> None:
    """The flows from example below have a similar structure: `set_slots` step is
    located third in both flows, which may lead to name collision - in both flows this
    step will get the name `2_set_slots` in both flows.
    In this case the `select_next_step` function could return the wrong step if the
    step ids are not unique.
    """
    flows_data = """
        flows:
          parent_flow:
            description: This is a test flow.
            steps:
              - call: child_flow
              - id: collect_foo_step
                collect: foo
              - set_slots:
                - slot_a: value_a
          child_flow:
            description: This is a test flow.
            steps:
              - collect: fizz
              - collect: buzz
              - set_slots:
                - abc: def
                - ghi: jkl
        """
    flows: FlowsList = YAMLFlowsReader.read_from_string(
        flows_data, file_path="path/flow.py"
    )
    mock_dialog_stack = Mock()
    mock_dialog_state_tracker = Mock()

    parent_flow = flows.flow_by_id("parent_flow")
    collect_foo_step = parent_flow.step_by_id("collect_foo_step")

    selected_next_step = select_next_step(
        collect_foo_step, parent_flow, mock_dialog_stack, mock_dialog_state_tracker
    )

    # if ids are not unique, `step_by_id` will return the first step with the given id
    # which in this case in from the called `child_flow`,
    # which is not the correct behavior
    assert selected_next_step.slots[0]["key"] == "slot_a"


def test_trigger_pattern_continue_interrupted_uses_localized_flow_name(
    monkeypatch: MonkeyPatch,
):
    # Load a flow with translations.
    german_flow_name = "German foo"
    flows = flows_from_str(
        f"""
        flows:
          foo:
            description: flow foo
            name: foo flow
            translation:
                de:
                  name: {german_flow_name}
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a tracker with a language slot set to German language.
    language = Language.from_language_code("de", is_default=True)
    slots = [
        StrictCategoricalSlot(
            "language", [], initial_value=language.code, values=[language.code]
        )
    ]
    tracker = DialogueStateTracker.from_events("test", [], slots=slots)

    # Create a stack with a flow frame.
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(flow_id="foo", step_id="first_step", frame_id="some-id")
        ]
    )
    tracker.update_stack(stack)

    # Trigger the pattern continue interrupted.
    current_frame = UserFlowStackFrame(
        flow_id="foo",
        step_id="second_step",
        frame_id="some-other-id",
        frame_type=FlowStackFrameType.INTERRUPT,
    )
    flow_executor.trigger_pattern_continue_interrupted(
        current_frame, stack, flows, tracker
    )

    # Confirm that the stack now contains a new frame with the localized flow name.
    top = stack.top()
    assert top is not None
    assert isinstance(top, ContinueInterruptedPatternFlowStackFrame)
    assert top.interrupted_flow_names == [german_flow_name]


@pytest.mark.asyncio
async def test_set_silence_timeout_at_step_collect():
    """Test that silence timeout is set correctly when running a collect step.

    We assess that event SlotSet is emitted with the correct silence timeout
    """
    silence_timeout = 10

    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
              silence_timeout: 10
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

    with structlog.testing.capture_logs() as caplog:
        result = await flow_executor.run_step(
            step,
            flow,
            stack,
            tracker,
            available_actions,
            flows,
            previous_step_id=START_STEP,
            slots=[],
        )

        logs = filter_logs(caplog, "flow.step.run.using_step_silence_timeout", "debug")

        assert len(logs) == 1

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, silence_timeout),
    ]
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[0], UserFlowStackFrame)
    assert isinstance(stack.frames[1], CollectInformationPatternFlowStackFrame)


@pytest.mark.asyncio
async def test_set_silence_timeout_at_step_collect_channel_specific(
    monkeypatch: MonkeyPatch,
):
    """Test that silence timeout is set correctly when running a collect step.

    We assess that event SlotSet is emitted with the correct silence timeout
    set for a specific channel.
    """
    silence_timeout = 10
    channel_name = "my_channel"

    flows = flows_from_str(
        f"""
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: collect_foo
              collect: foo
              silence_timeout:
                {channel_name}: {silence_timeout}
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    monkeypatch.setattr(tracker, "get_latest_input_channel", lambda: channel_name)

    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("collect_foo")

    available_actions = ["utter_ask_foo"]

    with structlog.testing.capture_logs() as caplog:
        result = await flow_executor.run_step(
            step,
            flow,
            stack,
            tracker,
            available_actions,
            flows,
            previous_step_id=START_STEP,
            slots=[],
        )

        logs = filter_logs(caplog, "flow.step.run.using_step_silence_timeout", "debug")

        assert len(logs) == 1

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, silence_timeout),
    ]
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[0], UserFlowStackFrame)
    assert isinstance(stack.frames[1], CollectInformationPatternFlowStackFrame)


@pytest.fixture
def interaction_handling_endpoint() -> InteractionHandlingConfig:
    return InteractionHandlingConfig(global_silence_timeout=10)


@pytest.fixture
def mock_available_agents(monkeypatch: MonkeyPatch) -> Iterator[MagicMock]:
    # Mock available agents accessed via Configuration singleton
    mock_available_agents_instance = MagicMock()
    mock_available_agents_instance.agents = {
        "car-research": {},
        "agent-1": {},
        "agent-2": {},
    }
    mock_available_agents_instance.get_agent_config.return_value = None

    mock_configuration_instance = MagicMock()
    mock_configuration_instance.available_agents = mock_available_agents_instance

    with patch(
        "rasa.core.config.configuration.Configuration.get_instance",
        return_value=mock_configuration_instance,
    ) as mock_method:
        yield mock_method


@pytest.mark.asyncio
async def test_reset_silence_timeout_to_global_at_step_collect(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that silence timeout is reset to the global value.

    We assess that event SlotSet is emitted with the global silence timeout configured
    in the interaction_handling.
    """
    # We set a global silence timeout in the interaction_handling endpoint
    global_silence_timeout = 11

    channel_name = "test_channel"

    channel_config = {
        SILENCE_TIMEOUT_CHANNEL_KEY: global_silence_timeout,
    }

    credentials = CredentialsConfig(
        channels={channel_name: channel_config}, config_file_path=Path()
    )

    monkeypatch.setattr(
        Configuration.get_instance(),
        "credentials",
        credentials,
    )

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

    # Emulate that customer had adjusted the global silence timeout

    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)

    monkeypatch.setattr(tracker, "get_latest_input_channel", lambda: channel_name)

    flow = flows.flow_by_id("my_flow")

    assert flow is not None
    step = flow.step_by_id("collect_foo")

    available_actions = ["utter_ask_foo"]

    with structlog.testing.capture_logs() as caplog:
        result = await flow_executor.run_step(
            step,
            flow,
            stack,
            tracker,
            available_actions,
            flows,
            previous_step_id=START_STEP,
            slots=[],
        )

        logs = filter_logs(caplog, "flow.step.run.use_channel_silence_timeout", "debug")

        assert len(logs) == 1

    assert isinstance(result, ContinueFlowWithNextStep)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, global_silence_timeout),
    ]
    assert len(stack.frames) == 2
    assert isinstance(stack.frames[0], UserFlowStackFrame)
    assert isinstance(stack.frames[1], CollectInformationPatternFlowStackFrame)


@pytest.mark.asyncio
async def test_append_global_silence_timeout_uses_channel_specific_value(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that silence timeout is set to the channel-specific value at flow start.

    When a flow starts with a non-collect step and the active channel has a
    silence_timeout configured in credentials, the SlotSet event should use
    the channel value rather than the global one.
    """
    channel_name = "test_channel"
    channel_silence_timeout = 5.0
    global_silence_timeout = 15.0

    credentials = CredentialsConfig(
        channels={channel_name: {SILENCE_TIMEOUT_CHANNEL_KEY: channel_silence_timeout}},
        config_file_path=Path(),
    )
    monkeypatch.setattr(
        Configuration.get_instance(),
        "credentials",
        credentials,
    )
    monkeypatch.setattr(
        Configuration.get_instance().endpoints,
        "interaction_handling",
        InteractionHandlingConfig(global_silence_timeout=global_silence_timeout),
    )

    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: first_action
              action: action_listen
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    monkeypatch.setattr(tracker, "get_latest_input_channel", lambda: channel_name)

    flow = flows.flow_by_id("my_flow")
    assert flow is not None
    step = flow.step_by_id("first_action")

    available_actions = ["action_listen"]

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, channel_silence_timeout),
    ]


@pytest.mark.asyncio
async def test_append_global_silence_timeout_when_channel_has_no_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test silence timeout is set to default when channel has no silence_timeout.

    When the active channel exists in credentials but has no silence_timeout key,
    the SlotSet event should use GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE.
    """
    channel_name = "test_channel"

    credentials = CredentialsConfig(
        channels={channel_name: {"some_other_key": "value"}},
        config_file_path=Path(),
    )
    monkeypatch.setattr(
        Configuration.get_instance(),
        "credentials",
        credentials,
    )

    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: first_action
              action: action_listen
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    monkeypatch.setattr(tracker, "get_latest_input_channel", lambda: channel_name)

    flow = flows.flow_by_id("my_flow")
    assert flow is not None
    step = flow.step_by_id("first_action")

    available_actions = ["action_listen"]

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE),
    ]


@pytest.mark.asyncio
async def test_append_global_silence_timeout_when_channel_not_in_credentials(
    monkeypatch: MonkeyPatch,
) -> None:
    """Test silence timeout is set to default when channel is not in credentials.

    When the tracker's input channel is not present in credentials at all,
    the function must not raise AttributeError and must fall back to
    GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE.
    """
    credentials = CredentialsConfig(
        channels={"some_other_channel": {SILENCE_TIMEOUT_CHANNEL_KEY: 5.0}},
        config_file_path=Path(),
    )
    monkeypatch.setattr(
        Configuration.get_instance(),
        "credentials",
        credentials,
    )

    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: first_action
              action: action_listen
        """
    )

    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    monkeypatch.setattr(tracker, "get_latest_input_channel", lambda: "unknown_channel")

    flow = flows.flow_by_id("my_flow")
    assert flow is not None
    step = flow.step_by_id("first_action")

    available_actions = ["action_listen"]

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.events == [
        FlowStarted(flow_id="my_flow"),
        SlotSet(SILENCE_TIMEOUT_SLOT, GLOBAL_SILENCE_TIMEOUT_DEFAULT_VALUE),
    ]


def test_set_dtmf_state_if_available_without_call_state():
    """Test that _set_dtmf_state_if_available handles missing call_state gracefully.

    This simulates the scenario where a collect step with DTMF config is run
    in a text channel (where call_state is not initialized). The function should
    not crash and should log a debug message.
    """

    step = CollectInformationFlowStep.from_json(
        "my_flow",
        {
            "collect": "account_number",
            "dtmf": {
                "length": 6,
                "allow_audio_input": False,
            },
        },
    )

    # Ensure call_state is not initialized (simulating text channel)
    # In this context, call_state should not be accessible
    with structlog.testing.capture_logs() as caplog:
        # This should not raise an exception
        flow_executor._set_dtmf_state_if_available(step)

        # Check that a debug log was created indicating the skip
        logs = filter_logs(caplog, "flow.step.run.collect.skip_dtmf", "debug")
        assert len(logs) == 1
        assert logs[0]["event_info"] == "call_state not initialized (non-voice channel)"


def test_set_dtmf_state_if_available_without_dtmf_config():
    step = CollectInformationFlowStep.from_json(
        "my_flow",
        {
            "collect": "account_number",
        },
    )

    # This should return early and not attempt to access call_state
    with structlog.testing.capture_logs() as caplog:
        flow_executor._set_dtmf_state_if_available(step)

        # No logs should be created since we return early
        logs = filter_logs(caplog, "flow.step.run.collect.skip_dtmf", "debug")
        assert len(logs) == 0


class TestGetParentUserFlowFrame:
    """Tests for _get_parent_user_flow_frame helper function."""

    def test_no_frames_returns_none(self):
        """Test that an empty stack returns None."""
        stack = DialogueStack(frames=[])
        assert flow_executor._get_parent_user_flow_frame(stack) is None

    def test_single_user_frame_returns_none(self):
        """Test that a stack with only one user flow frame returns None."""
        stack = DialogueStack(
            frames=[
                UserFlowStackFrame(
                    flow_id="flow_a",
                    step_id="step_1",
                    frame_type=FlowStackFrameType.REGULAR,
                )
            ]
        )
        assert flow_executor._get_parent_user_flow_frame(stack) is None

    def test_two_user_frames_returns_parent(self):
        """Test that a stack with two user flow frames returns the parent."""
        parent_frame = UserFlowStackFrame(
            flow_id="parent_flow",
            step_id="call_step",
            frame_type=FlowStackFrameType.REGULAR,
        )
        child_frame = UserFlowStackFrame(
            flow_id="child_flow",
            step_id="first_step",
            frame_type=FlowStackFrameType.CALL,
        )
        stack = DialogueStack(frames=[parent_frame, child_frame])

        result = flow_executor._get_parent_user_flow_frame(stack)

        assert result == parent_frame
        assert result.flow_id == "parent_flow"
        assert result.step_id == "call_step"

    def test_mixed_frames_returns_correct_parent(self):
        """Test that non-user frames are ignored when finding parent."""
        parent_frame = UserFlowStackFrame(
            flow_id="parent_flow",
            step_id="collect_step",
            frame_type=FlowStackFrameType.REGULAR,
        )
        pattern_frame = CollectInformationPatternFlowStackFrame(
            collect="my_slot",
            utter="utter_ask_my_slot",
        )
        child_frame = UserFlowStackFrame(
            flow_id="child_flow",
            step_id="first_step",
            frame_type=FlowStackFrameType.CALL,
        )
        stack = DialogueStack(frames=[parent_frame, pattern_frame, child_frame])

        result = flow_executor._get_parent_user_flow_frame(stack)

        # Should return parent_frame, ignoring the pattern frame
        assert result == parent_frame
        assert result.flow_id == "parent_flow"

    def test_nested_calls_returns_immediate_parent(self):
        """Test that with multiple nested calls, the immediate parent is returned."""
        grandparent_frame = UserFlowStackFrame(
            flow_id="grandparent_flow",
            step_id="call_parent",
            frame_type=FlowStackFrameType.REGULAR,
        )
        parent_frame = UserFlowStackFrame(
            flow_id="parent_flow",
            step_id="call_child",
            frame_type=FlowStackFrameType.CALL,
        )
        child_frame = UserFlowStackFrame(
            flow_id="child_flow",
            step_id="first_step",
            frame_type=FlowStackFrameType.CALL,
        )
        stack = DialogueStack(frames=[grandparent_frame, parent_frame, child_frame])

        result = flow_executor._get_parent_user_flow_frame(stack)

        # Should return parent_frame (immediate parent of child_frame)
        assert result == parent_frame
        assert result.flow_id == "parent_flow"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason, expected_action",
    [
        (
            RASA_PATTERN_CANNOT_HANDLE_NO_RELEVANT_ANSWER,
            "utter_no_relevant_answer_found",
        ),
        (
            "cannot_handle_default",
            "utter_ask_rephrase",
        ),
    ],
    ids=["no_relevant_answer", "default_reason"],
)
async def test_pattern_cannot_handle_routes_by_reason(
    reason: str,
    expected_action: str,
) -> None:
    """Test that pattern_cannot_handle routes to the correct utterance.

    When context.reason is 'cannot_handle_no_relevant_answer'
    (e.g. enterprise search returns no documents), the flow should select
    'utter_no_relevant_answer_found'. For the default reason it should
    fall back to 'utter_ask_rephrase'.
    """
    # Load the real default pattern flows so pattern_cannot_handle is available
    all_flows = flows_from_str_including_defaults(
        """
        flows:
          dummy_flow:
            description: placeholder
            steps:
            - id: "1"
              action: utter_hello
        """
    )

    stack = DialogueStack(
        frames=[
            CannotHandlePatternFlowStackFrame(reason=reason),
        ]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen")],
    )
    tracker.update_stack(stack)

    available_actions = [
        "utter_ask_rephrase",
        "utter_no_relevant_answer_found",
        "utter_cannot_handle",
    ]

    prediction = await flow_executor.advance_flows(
        tracker, available_actions, all_flows, slots=[]
    )

    assert prediction.action_name == expected_action


@pytest.mark.asyncio
async def test_run_step_link_customer_satisfaction():
    """Test that a user flow can link to pattern_customer_satisfaction."""
    flows = flows_from_str_including_defaults(
        """
        flows:
          my_flow:
            description: flow my_flow
            steps:
            - id: link
              link: pattern_customer_satisfaction
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
    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions,
        flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, ContinueFlowWithNextStep)

    top = stack.top()
    assert isinstance(top, UserFlowStackFrame)
    assert top.flow_id == "my_flow"

    linked_flow = stack.frames[0]
    assert isinstance(linked_flow, UserFlowStackFrame)
    assert linked_flow.flow_id == "pattern_customer_satisfaction"
    assert linked_flow.frame_type == FlowStackFrameType.LINK


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csat_score_value,expected_response",
    [
        ("satisfied", "utter_csat_thank_you_satisfied"),
        ("unsatisfied", "utter_csat_thank_you_unsatisfied"),
    ],
)
async def test_pattern_customer_satisfaction_full_flow(
    csat_score_value: str, expected_response: str
):
    """Test the full CSAT flow: asks for score, then thanks user after collection."""
    flows = flows_from_str_including_defaults(
        """
        flows:
          dummy_flow:
            description: placeholder
            steps:
            - id: "1"
              action: utter_hello
        """
    )

    csat_slot = StrictCategoricalSlot(
        name="csat_score",
        mappings=[],
        values=["satisfied", "unsatisfied"],
    )

    available_actions = [
        "utter_ask_csat_score",
        "utter_csat_thank_you_satisfied",
        "utter_csat_thank_you_unsatisfied",
    ]

    # Step 1: Start CSAT flow - should ask for score
    csat_frame = CustomerSatisfactionPatternFlowStackFrame(
        frame_id="csat-frame-id",
        step_id=START_STEP,
    )
    stack = DialogueStack(frames=[csat_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen")],
        slots=[csat_slot],
    )
    tracker.update_stack(stack)

    prediction = await flow_executor.advance_flows(
        tracker, available_actions, flows, slots=[csat_slot]
    )
    assert prediction.action_name == "utter_ask_csat_score"

    # Apply events so tracker gets the updated stack (collect frame was pushed)
    tracker.update_with_events(prediction.events or [])

    # Step 2: User provides score  and assistant should respond appropriately
    tracker.update(SlotSet("csat_score", csat_score_value))
    tracker.update(ActionExecuted(action_name="utter_ask_csat_score"))
    tracker.update(ActionExecuted(action_name="action_listen"))

    prediction = await flow_executor.advance_flows(
        tracker, available_actions, flows, slots=[csat_slot]
    )
    assert prediction.action_name == expected_response


@pytest.mark.asyncio
async def test_pattern_completed_links_to_customer_satisfaction():
    """Test that pattern_completed links to pattern_customer_satisfaction."""
    flows = flows_from_str_including_defaults(
        """
        flows:
          my_flow:
            description: test flow
            steps:
            - id: "1"
              action: utter_hello
        """
    )

    continue_conversation_slot = BooleanSlot(
        name="continue_conversation",
        mappings=[],
        initial_value=None,
    )

    csat_slot = StrictCategoricalSlot(
        name="csat_score",
        mappings=[],
        values=["satisfied", "unsatisfied"],
    )

    available_actions = [
        "utter_ask_continue_conversation",
        "utter_closing_words",
        "utter_can_do_something_else",
        "utter_ask_csat_score",
    ]

    # Start with pattern_completed frame (user declined to continue)
    completed_frame = CompletedPatternFlowStackFrame(
        frame_id="completed-frame-id",
        step_id=START_STEP,
        previous_flow_name="my_flow",
    )
    stack = DialogueStack(frames=[completed_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[ActionExecuted(action_name="action_listen")],
        slots=[continue_conversation_slot, csat_slot],
    )
    tracker.update_stack(stack)

    # First advance should ask about continuing conversation
    prediction = await flow_executor.advance_flows(
        tracker,
        available_actions,
        flows,
        slots=[continue_conversation_slot, csat_slot],
    )
    assert prediction.action_name == "utter_ask_continue_conversation"

    # Apply events and set user's response to NOT continue
    tracker.update_with_events(prediction.events or [])
    tracker.update(SlotSet("continue_conversation", False))
    tracker.update(ActionExecuted(action_name="utter_ask_continue_conversation"))
    tracker.update(ActionExecuted(action_name="action_listen"))

    # Next advance should say closing words
    prediction = await flow_executor.advance_flows(
        tracker,
        available_actions,
        flows,
        slots=[continue_conversation_slot, csat_slot],
    )
    assert prediction.action_name == "utter_closing_words"

    # Apply events and advance again - should now link to CSAT and ask for score
    tracker.update_with_events(prediction.events or [])
    tracker.update(ActionExecuted(action_name="utter_closing_words"))
    tracker.update(ActionExecuted(action_name="action_listen"))

    prediction = await flow_executor.advance_flows(
        tracker,
        available_actions,
        flows,
        slots=[continue_conversation_slot, csat_slot],
    )
    assert prediction.action_name == "utter_ask_csat_score"


async def test_executor_exits_gracefully_when_tracker_is_terminated():
    """Test that flow executor exits gracefully when tracker is terminated.

    When a SessionEnded event has been applied to the tracker,
    the flow executor should exit the loop and return
    action_listen instead of continuing to process steps
    (which would cause an infinite loop as events are ignored).
    """
    flow_with_action = flows_from_str(
        """
        flows:
          foo_flow:
            description: flow foo
            steps:
            - id: "1"
              action: action_foo
            - id: "2"
              action: action_bar
              next: END
        """
    )

    domain = Domain.empty()

    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            ActionExecuted(action_name="action_listen"),
            SessionEnded(),
        ],
        domain=domain,
        slots=domain.slots,
    )
    tracker.update_stack(stack)

    assert tracker.terminated is True

    available_actions = ["action_foo", "action_bar", "action_listen"]

    # The flow executor should exit gracefully with action_listen
    # instead of raising FlowCircuitBreakerTrippedException
    selection = await flow_executor.advance_flows_until_next_action(
        tracker, available_actions, flow_with_action, slots=[]
    )

    assert selection.action_name == "action_listen"
    assert selection.score == 1.0


# ---------------------------------------------------------------------------
# action_session_start skip-in-flow tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_step_skips_action_session_start_when_session_already_started():
    """run_step returns ContinueFlowWithNextStep (no pause) when action_session_start
    is the current step and the session was already started for this message."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow with action_session_start step
            steps:
            - id: session_step
              action: action_session_start
            - id: next_step
              action: utter_greet
        """
    )
    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("session_step")

    tracker.has_session_started_for_current_message = Mock(return_value=True)

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions=[ACTION_SESSION_START_NAME, "utter_greet"],
        flows=flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, ContinueFlowWithNextStep)


@pytest.mark.asyncio
async def test_run_step_does_not_skip_action_session_start_when_session_not_started():
    """run_step returns PauseFlowReturnPrediction (normal path) when
    action_session_start is the current step but the session has not
    been started yet for this message."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow with action_session_start step
            steps:
            - id: session_step
              action: action_session_start
        """
    )
    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("session_step")

    tracker.has_session_started_for_current_message = Mock(return_value=False)

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions=[ACTION_SESSION_START_NAME],
        flows=flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == ACTION_SESSION_START_NAME


@pytest.mark.asyncio
async def test_advance_flows_skips_action_session_start_and_returns_next_action():
    """advance_flows_until_next_action skips an action_session_start step when the
    session is already started and returns the next flow step's action instead of
    falling back to action_listen."""
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow starting with action_session_start
            steps:
            - id: session_step
              action: action_session_start
            - id: greet_step
              action: utter_greet
        """
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="my_flow", step_id="START", frame_id="some-frame-id"
            )
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(stack)

    # Simulate: session was already started for the current message.
    tracker.has_session_started_for_current_message = Mock(return_value=True)

    available_actions = [ACTION_SESSION_START_NAME, "utter_greet"]
    prediction = await flow_executor.advance_flows_until_next_action(
        tracker, available_actions, flows, slots=[]
    )

    assert prediction.action_name == "utter_greet", (
        "Expected flow to skip action_session_start and advance to utter_greet, "
        f"but got '{prediction.action_name}'"
    )


@pytest.mark.asyncio
async def test_run_step_does_not_skip_non_session_start_action_when_session_started():
    """run_step does not skip a regular action even when session is already started.

    The skip guard is specifically scoped to action_session_start. Any other
    ActionFlowStep must still produce a PauseFlowReturnPrediction regardless of
    the session state.
    """
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow with a regular action step
            steps:
            - id: greet_step
              action: utter_greet
        """
    )
    user_flow_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="START", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_flow_frame])
    tracker = DialogueStateTracker.from_events("test", [])
    tracker.update_stack(stack)
    flow = flows.flow_by_id("my_flow")
    step = flow.step_by_id("greet_step")

    # Session is already started — must NOT affect non-session-start steps.
    tracker.has_session_started_for_current_message = Mock(return_value=True)

    result = await flow_executor.run_step(
        step,
        flow,
        stack,
        tracker,
        available_actions=["utter_greet"],
        flows=flows,
        previous_step_id=START_STEP,
        slots=[],
    )

    assert isinstance(result, PauseFlowReturnPrediction)
    assert result.action_prediction.action_name == "utter_greet"


@pytest.mark.asyncio
async def test_advance_flows_with_only_action_session_start_step_skips_and_ends():
    """advance_flows_until_next_action returns action_listen when a flow
    contains only an action_session_start step and that step is skipped
    due to session already started.

    The flow should complete gracefully (via EndFlowStep) rather than stalling.
    """
    flows = flows_from_str(
        """
        flows:
          my_flow:
            description: single-step flow with action_session_start
            steps:
            - id: session_step
              action: action_session_start

          pattern_completed:
             description: wraps up a completed user flow
             steps:
                - action: utter_what_else_can_i_do
        """
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="my_flow", step_id="START", frame_id="some-frame-id"
            )
        ]
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(stack)

    tracker.has_session_started_for_current_message = Mock(return_value=True)

    available_actions = [ACTION_SESSION_START_NAME, "action_listen"]
    prediction = await flow_executor.advance_flows_until_next_action(
        tracker, available_actions, flows, slots=[]
    )

    # Flow ends with no real action to predict → falls back to action_listen.
    assert prediction.action_name == "action_listen", (
        "Expected action_listen after flow ends with only a skipped "
        f"action_session_start step, but got '{prediction.action_name}'"
    )
