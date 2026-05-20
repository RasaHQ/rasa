from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT,
    FLOW_PATTERN_CONTINUE_INTERRUPTED,
    INTERRUPTED_FLOW_TO_CONTINUE_SLOT,
    ActionCancelInterruptedFlows,
    ActionContinueInterruptedFlow,
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.shared.core.events import AgentResumed, FlowCancelled, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


class TestContinueInterruptedPatternFlowStackFrame:
    """Test the ContinueInterruptedPatternFlowStackFrame class."""

    def test_default_initialization(self):
        """Test default initialization of the frame."""
        frame = ContinueInterruptedPatternFlowStackFrame()

        assert frame.flow_id == FLOW_PATTERN_CONTINUE_INTERRUPTED
        assert frame.interrupted_flow_names == []
        assert frame.interrupted_flow_ids == []
        assert frame.interrupted_flow_options == ""
        assert frame.multiple_flows_interrupted is False
        assert frame.step_id == "START"  # inherited from BaseFlowStackFrame

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        frame = ContinueInterruptedPatternFlowStackFrame(
            frame_id="test_frame",
            step_id="test_step",
            interrupted_flow_names=["flow1", "flow2"],
            interrupted_flow_ids=["id1", "id2"],
            interrupted_flow_options="flow1, flow2",
            multiple_flows_interrupted=True,
        )

        assert frame.frame_id == "test_frame"
        assert frame.step_id == "test_step"
        assert frame.interrupted_flow_names == ["flow1", "flow2"]
        assert frame.interrupted_flow_ids == ["id1", "id2"]
        assert frame.interrupted_flow_options == "flow1, flow2"
        assert frame.multiple_flows_interrupted is True

    def test_type_method(self):
        """Test the type class method."""
        frame = ContinueInterruptedPatternFlowStackFrame()
        assert frame.type() == FLOW_PATTERN_CONTINUE_INTERRUPTED

    def test_from_dict_with_all_fields(self):
        """Test from_dict method with all fields present."""
        data = {
            "frame_id": "test_id",
            "step_id": "test_step",
            "interrupted_flow_names": ["flow1", "flow2"],
            "interrupted_flow_ids": ["id1", "id2"],
            "interrupted_flow_options": "flow1, flow2",
            "multiple_flows_interrupted": True,
        }

        frame = ContinueInterruptedPatternFlowStackFrame.from_dict(data)

        assert frame.frame_id == "test_id"
        assert frame.step_id == "test_step"
        assert frame.interrupted_flow_names == ["flow1", "flow2"]
        assert frame.interrupted_flow_ids == ["id1", "id2"]
        assert frame.interrupted_flow_options == "flow1, flow2"
        assert frame.multiple_flows_interrupted is True

    def test_from_dict_with_minimal_fields(self):
        """Test from_dict method with minimal required fields."""
        data = {
            "frame_id": "test_id",
            "step_id": "test_step",
            "interrupted_flow_names": [],
            "interrupted_flow_ids": [],
            "interrupted_flow_options": "",
        }

        frame = ContinueInterruptedPatternFlowStackFrame.from_dict(data)

        assert frame.frame_id == "test_id"
        assert frame.step_id == "test_step"
        assert frame.interrupted_flow_names == []
        assert frame.interrupted_flow_ids == []
        assert frame.interrupted_flow_options == ""
        assert frame.multiple_flows_interrupted is False

    def test_from_dict_calculates_multiple_flows_interrupted(self):
        """Test that from_dict calculates multiple_flows_interrupted correctly."""
        # Single flow
        data_single = {
            "frame_id": "test_id",
            "step_id": "test_step",
            "interrupted_flow_names": ["flow1"],
            "interrupted_flow_ids": ["id1"],
            "interrupted_flow_options": "flow1",
        }
        frame_single = ContinueInterruptedPatternFlowStackFrame.from_dict(data_single)
        assert frame_single.multiple_flows_interrupted is False

        # Multiple flows
        data_multiple = {
            "frame_id": "test_id",
            "step_id": "test_step",
            "interrupted_flow_names": ["flow1", "flow2"],
            "interrupted_flow_ids": ["id1", "id2"],
            "interrupted_flow_options": "flow1 and flow2",
        }
        frame_multiple = ContinueInterruptedPatternFlowStackFrame.from_dict(
            data_multiple
        )
        assert frame_multiple.multiple_flows_interrupted is True

    def test_equality(self):
        """Test equality comparison between frames."""
        frame1 = ContinueInterruptedPatternFlowStackFrame(
            step_id="step1",
            interrupted_flow_names=["flow1"],
            interrupted_flow_ids=["id1"],
        )

        frame2 = ContinueInterruptedPatternFlowStackFrame(
            step_id="step1",
            interrupted_flow_names=["flow1"],
            interrupted_flow_ids=["id1"],
        )

        frame3 = ContinueInterruptedPatternFlowStackFrame(
            step_id="step2",
            interrupted_flow_names=["flow1"],
            interrupted_flow_ids=["id1"],
        )

        assert frame1 == frame2
        assert frame1 != frame3
        assert frame1 != "not a frame"

    def test_inheritance(self):
        """Test that the frame inherits from the correct base class."""
        frame = ContinueInterruptedPatternFlowStackFrame()
        assert isinstance(frame, PatternFlowStackFrame)


class TestActionContinueInterruptedFlow:
    """Test the ActionContinueInterruptedFlow class."""

    def test_name(self):
        """Test the action name."""
        action = ActionContinueInterruptedFlow()
        assert action.name() == "action_continue_interrupted_flow"

    @pytest.mark.asyncio
    async def test_run_no_pattern_frame(self):
        """Test run when no pattern frame is found."""
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock the utility function to return None
        with patch(
            "rasa.dialogue_understanding.patterns.continue_interrupted."
            "get_active_pattern_frame"
        ) as mock_get:
            mock_get.return_value = None

            action = ActionContinueInterruptedFlow()
            events = await action.run(output_channel, nlg, tracker, domain)

            assert events == []
            mock_get.assert_called_once_with(
                tracker.stack, ContinueInterruptedPatternFlowStackFrame
            )

    @pytest.mark.asyncio
    async def test_run_single_flow_interrupted(self):
        """Test run when a single flow was interrupted."""
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock pattern frame
        pattern_frame = ContinueInterruptedPatternFlowStackFrame()
        pattern_frame.interrupted_flow_ids = ["flow1"]
        pattern_frame.interrupted_flow_names = ["Flow 1"]
        pattern_frame.multiple_flows_interrupted = False

        # Mock utility functions
        with (
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted."
                "get_active_pattern_frame"
            ) as mock_get,
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted.resume_flow"
            ) as mock_resume,
        ):
            mock_get.return_value = pattern_frame
            mock_resume.return_value = [MagicMock()]

            action = ActionContinueInterruptedFlow()
            events = await action.run(output_channel, nlg, tracker, domain)

            # Should call resume_flow with the first interrupted flow
            mock_resume.assert_called_once_with("flow1", tracker, tracker.stack)

            # Should return resume events plus slot clearing events
            assert len(events) == 3  # resume event + 2 slot clearing events
            assert any(
                isinstance(e, SlotSet) and e.key == INTERRUPTED_FLOW_TO_CONTINUE_SLOT
                for e in events
            )
            assert any(
                isinstance(e, SlotSet)
                and e.key == CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT
                for e in events
            )

    @pytest.mark.asyncio
    async def test_run_multiple_flows_interrupted_valid_selection(self):
        """Test run when multiple flows were interrupted and user selects valid flow."""
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock pattern frame
        pattern_frame = ContinueInterruptedPatternFlowStackFrame()
        pattern_frame.interrupted_flow_ids = ["flow1", "flow2"]
        pattern_frame.interrupted_flow_names = ["Flow 1", "Flow 2"]
        pattern_frame.multiple_flows_interrupted = True

        # Mock tracker to return a valid flow selection
        tracker.get_slot.return_value = "flow1"

        # Mock utility functions
        with (
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted."
                "get_active_pattern_frame"
            ) as mock_get,
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted.resume_flow"
            ) as mock_resume,
        ):
            mock_get.return_value = pattern_frame
            mock_resume.return_value = [MagicMock()]

            action = ActionContinueInterruptedFlow()
            events = await action.run(output_channel, nlg, tracker, domain)

            # Should call resume_flow with the selected flow
            mock_resume.assert_called_once_with("flow1", tracker, tracker.stack)

            # Should return resume events plus slot clearing events
            assert len(events) == 3  # resume event + 2 slot clearing events

    @pytest.mark.asyncio
    async def test_run_multiple_flows_interrupted_flow_name_selection(self):
        """Test run when multiple flows were interrupted and user selects by
        flow name.
        """
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock pattern frame
        pattern_frame = ContinueInterruptedPatternFlowStackFrame()
        pattern_frame.interrupted_flow_ids = ["flow1", "flow2"]
        pattern_frame.interrupted_flow_names = ["Flow 1", "Flow 2"]
        pattern_frame.multiple_flows_interrupted = True

        # Mock tracker to return a flow name (not ID)
        tracker.get_slot.return_value = "Flow 2"

        # Mock utility functions
        with (
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted."
                "get_active_pattern_frame"
            ) as mock_get,
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted.resume_flow"
            ) as mock_resume,
        ):
            mock_get.return_value = pattern_frame
            mock_resume.return_value = [MagicMock()]

            action = ActionContinueInterruptedFlow()
            events = await action.run(output_channel, nlg, tracker, domain)

            # Should call resume_flow with the flow ID corresponding to the name
            mock_resume.assert_called_once_with("flow2", tracker, tracker.stack)

            # Should return resume events plus slot clearing events
            assert len(events) == 3  # resume event + 2 slot clearing events

    @pytest.mark.asyncio
    async def test_run_multiple_flows_interrupted_invalid_selection(self):
        """Test run when multiple flows were interrupted and user selects
        invalid flow.
        """
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = AsyncMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock pattern frame
        pattern_frame = ContinueInterruptedPatternFlowStackFrame()
        pattern_frame.interrupted_flow_ids = ["flow1", "flow2"]
        pattern_frame.interrupted_flow_names = ["Flow 1", "Flow 2"]
        pattern_frame.multiple_flows_interrupted = True

        # Mock tracker to return an invalid flow selection
        tracker.get_slot.return_value = "invalid_flow"

        # Mock utility functions
        with patch(
            "rasa.dialogue_understanding.patterns.continue_interrupted."
            "get_active_pattern_frame"
        ) as mock_get:
            mock_get.return_value = pattern_frame

            action = ActionContinueInterruptedFlow()
            events = await action.run(output_channel, nlg, tracker, domain)

            # Should send a message asking for valid selection
            output_channel.send_text_message.assert_called_once()
            call_args = output_channel.send_text_message.call_args[0]
            assert "You haven't selected a valid task to resume" in call_args[1]

            # Should return no events
            assert events == []


class TestActionCancelInterruptedFlows:
    """Test the ActionCancelInterruptedFlows class."""

    def test_name(self):
        """Test the action name."""
        action = ActionCancelInterruptedFlows()
        assert action.name() == "action_cancel_interrupted_flows"

    @pytest.mark.asyncio
    async def test_run_no_pattern_frame(self):
        """Test run when no pattern frame is found."""
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock the utility function to return None
        with patch(
            "rasa.dialogue_understanding.patterns.continue_interrupted."
            "get_active_pattern_frame"
        ) as mock_get:
            mock_get.return_value = None

            action = ActionCancelInterruptedFlows()
            events = await action.run(output_channel, nlg, tracker, domain)

            assert events == []
            mock_get.assert_called_once_with(
                tracker.stack, ContinueInterruptedPatternFlowStackFrame
            )

    @pytest.mark.asyncio
    async def test_run_with_interrupted_flows(self):
        """Test run when there are interrupted flows to cancel."""
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock pattern frame
        pattern_frame = ContinueInterruptedPatternFlowStackFrame()
        pattern_frame.interrupted_flow_ids = ["flow1", "flow2"]

        # Mock utility functions
        with (
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted."
                "get_active_pattern_frame"
            ) as mock_get,
            patch(
                "rasa.dialogue_understanding.patterns.continue_interrupted.ActionCancelInterruptedFlows.cancel_flow"
            ) as mock_cancel,
        ):
            mock_get.return_value = pattern_frame
            mock_cancel.side_effect = lambda t, s, f: [
                MagicMock(key=f"cancel_event_{f}")
            ]

            action = ActionCancelInterruptedFlows()
            events = await action.run(output_channel, nlg, tracker, domain)

            # Should call cancel_flow for each interrupted flow
            assert mock_cancel.call_count == 2
            mock_cancel.assert_any_call(tracker, tracker.stack, "flow1")
            mock_cancel.assert_any_call(tracker, tracker.stack, "flow2")

            # Should return cancel events plus slot clearing events
            assert len(events) == 4  # 2 cancel events + 2 slot clearing events
            assert any(
                isinstance(e, SlotSet) and e.key == INTERRUPTED_FLOW_TO_CONTINUE_SLOT
                for e in events
            )
            assert any(
                isinstance(e, SlotSet)
                and e.key == CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT
                for e in events
            )

    @pytest.mark.asyncio
    async def test_run_no_interrupted_flows(self):
        """Test run when there are no interrupted flows."""
        tracker = MagicMock()
        tracker.stack = MagicMock()
        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        # Mock pattern frame
        pattern_frame = ContinueInterruptedPatternFlowStackFrame()
        pattern_frame.interrupted_flow_ids = []

        # Mock utility functions
        with patch(
            "rasa.dialogue_understanding.patterns.continue_interrupted."
            "get_active_pattern_frame"
        ) as mock_get:
            mock_get.return_value = pattern_frame

            action = ActionCancelInterruptedFlows()
            events = await action.run(output_channel, nlg, tracker, domain)

            # Should return only slot clearing events
            assert len(events) == 2  # 2 slot clearing events
            assert any(
                isinstance(e, SlotSet) and e.key == INTERRUPTED_FLOW_TO_CONTINUE_SLOT
                for e in events
            )
            assert any(
                isinstance(e, SlotSet)
                and e.key == CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT
                for e in events
            )

    @pytest.mark.parametrize(
        "flow_id, stack_frames, expected_events_count, expected_cancel_frame",
        [
            # Test case 1: Cancel a flow that exists on the stack
            (
                "flow_1",
                [
                    UserFlowStackFrame(
                        flow_id="flow_1",
                        step_id="START",
                        frame_id="user-frame-1",
                        frame_type=FlowStackFrameType.REGULAR,
                    ),
                    AgentStackFrame(
                        frame_id="agent-frame-1",
                        state=AgentState.WAITING_FOR_INPUT,
                        agent_id="test_agent",
                        flow_id="flow_1",
                    ),
                ],
                3,  # FlowCancelled + AgentCancelled + stack update events
                True,  # Cancel frame should be added
            ),
            # Test case 2: Cancel a flow that doesn't exist on the stack
            (
                "nonexistent_flow",
                [
                    UserFlowStackFrame(
                        flow_id="flow_1",
                        step_id="START",
                        frame_id="user-frame-1",
                        frame_type=FlowStackFrameType.REGULAR,
                    ),
                ],
                0,  # No events should be created
                False,  # No cancel frame should be added
            ),
            # Test case 3: Cancel a flow with multiple frames
            (
                "flow_2",
                [
                    UserFlowStackFrame(
                        flow_id="flow_1",
                        step_id="START",
                        frame_id="user-frame-1",
                        frame_type=FlowStackFrameType.REGULAR,
                    ),
                    UserFlowStackFrame(
                        flow_id="flow_2",
                        step_id="START",
                        frame_id="user-frame-2",
                        frame_type=FlowStackFrameType.REGULAR,
                    ),
                    AgentStackFrame(
                        frame_id="agent-frame-2",
                        state=AgentState.WAITING_FOR_INPUT,
                        agent_id="test_agent",
                        flow_id="flow_2",
                    ),
                ],
                3,  # FlowCancelled + AgentCancelled + stack update events
                True,  # Cancel frame should be added
            ),
            # Test case 4: Cancel an interrupted flow
            (
                "interrupted_flow",
                [
                    UserFlowStackFrame(
                        flow_id="interrupted_flow",
                        step_id="START",
                        frame_id="user-frame-1",
                        frame_type=FlowStackFrameType.INTERRUPT,
                    ),
                    AgentStackFrame(
                        frame_id="agent-frame-1",
                        state=AgentState.WAITING_FOR_INPUT,
                        agent_id="test_agent",
                        flow_id="interrupted_flow",
                    ),
                ],
                3,  # FlowCancelled + AgentCancelled + stack update events
                True,  # Cancel frame should be added
            ),
        ],
    )
    def test_cancel_flow(
        self,
        flow_id: str,
        stack_frames: List[DialogueStackFrame],
        expected_events_count: int,
        expected_cancel_frame: bool,
    ):
        """Test cancel_flow function with various stack configurations."""
        # Given
        stack = DialogueStack(frames=stack_frames)
        tracker = DialogueStateTracker.from_events("test_sender", [])

        # When
        action = ActionCancelInterruptedFlows()
        events = action.cancel_flow(tracker, stack, flow_id)

        # Then
        assert len(events) == expected_events_count

        if expected_cancel_frame:
            # Check that a cancel frame was added to the stack
            cancel_frame = stack.top()
            assert isinstance(cancel_frame, CancelPatternFlowStackFrame)
            assert cancel_frame.canceled_name == flow_id

            # Check that FlowCancelled event was created
            flow_cancelled_events = [e for e in events if isinstance(e, FlowCancelled)]
            assert len(flow_cancelled_events) == 1
            flow_cancelled_event = flow_cancelled_events[0]
            assert flow_cancelled_event.flow_id == flow_id

            # Check that stack update events were created
            stack_update_events = [
                e for e in events if not isinstance(e, FlowCancelled)
            ]
            assert len(stack_update_events) > 0
        else:
            # Check that no cancel frame was added
            assert not isinstance(stack.top(), CancelPatternFlowStackFrame)

    def test_cancel_flow_preserves_original_stack(self):
        """Test that cancel_flow doesn't modify the original stack structure."""
        # Given
        original_frames = [
            UserFlowStackFrame(
                flow_id="flow_1",
                step_id="START",
                frame_id="user-frame-1",
                frame_type=FlowStackFrameType.REGULAR,
            ),
            AgentStackFrame(
                frame_id="agent-frame-1",
                state=AgentState.WAITING_FOR_INPUT,
                agent_id="test_agent",
                flow_id="flow_1",
            ),
        ]
        original_stack = DialogueStack(frames=original_frames.copy())
        tracker = DialogueStateTracker.from_events("test_sender", [])

        # When
        action = ActionCancelInterruptedFlows()
        action.cancel_flow(tracker, original_stack, "flow_1")

        # Then
        # The original frames should still be there
        assert len(original_stack.frames) >= len(original_frames)
        for i, frame in enumerate(original_frames):
            assert original_stack.frames[i] == frame

    def test_cancel_flow_with_empty_stack(self):
        """Test cancel_flow with an empty stack."""
        # Given
        empty_stack = DialogueStack(frames=[])
        tracker = DialogueStateTracker.from_events("test_sender", [])

        # When
        action = ActionCancelInterruptedFlows()
        events = action.cancel_flow(tracker, empty_stack, "any_flow")

        # Then
        assert len(events) == 0
        assert len(empty_stack.frames) == 0

    def test_cancel_flow_collects_correct_frames(self):
        """Test that cancel_flow collects the correct frames to cancel."""
        # Given
        stack_frames: List[DialogueStackFrame] = [
            UserFlowStackFrame(
                flow_id="flow_1",
                step_id="START",
                frame_id="user-frame-1",
                frame_type=FlowStackFrameType.REGULAR,
            ),
            AgentStackFrame(
                frame_id="agent-frame-1",
                state=AgentState.WAITING_FOR_INPUT,
                agent_id="test_agent",
                flow_id="flow_1",
            ),
            UserFlowStackFrame(
                flow_id="flow_2",
                step_id="START",
                frame_id="user-frame-2",
                frame_type=FlowStackFrameType.REGULAR,
            ),
            AgentStackFrame(
                frame_id="agent-frame-2",
                state=AgentState.WAITING_FOR_INPUT,
                agent_id="test_agent",
                flow_id="flow_2",
            ),
        ]
        stack = DialogueStack(frames=stack_frames)
        tracker = DialogueStateTracker.from_events("test_sender", [])

        # When
        action = ActionCancelInterruptedFlows()
        events = action.cancel_flow(tracker, stack, "flow_2")

        # Then
        assert len(events) == 3  # FlowCancelled + AgentCancelled + stack update events

        # Check that the cancel frame contains the correct frame IDs
        cancel_frame = stack.top()
        assert isinstance(cancel_frame, CancelPatternFlowStackFrame)
        assert cancel_frame.canceled_name == "flow_2"

        # Should include flow_2 and agent-frame-2
        expected_canceled_frames = ["user-frame-2", "agent-frame-2"]
        assert set(cancel_frame.canceled_frames) == set(expected_canceled_frames)


class TestIntegration:
    """Integration tests for the continue_interrupted module."""

    def test_frame_serialization_roundtrip(self):
        """Test that a frame can be serialized and deserialized correctly."""
        original_frame = ContinueInterruptedPatternFlowStackFrame(
            frame_id="test_frame",
            step_id="test_step",
            interrupted_flow_names=["flow1", "flow2"],
            interrupted_flow_ids=["id1", "id2"],
            interrupted_flow_options="flow1, flow2",
            multiple_flows_interrupted=True,
        )

        # Serialize to dict
        frame_dict = original_frame.as_dict()

        # Deserialize from dict
        restored_frame = ContinueInterruptedPatternFlowStackFrame.from_dict(frame_dict)

        # Should be equal
        assert original_frame == restored_frame

        # Check specific fields
        assert restored_frame.frame_id == "test_frame"
        assert restored_frame.step_id == "test_step"
        assert restored_frame.interrupted_flow_names == ["flow1", "flow2"]
        assert restored_frame.interrupted_flow_ids == ["id1", "id2"]
        assert restored_frame.interrupted_flow_options == "flow1, flow2"
        assert restored_frame.multiple_flows_interrupted is True

    def test_frame_type_consistency(self):
        """Test that the frame type is consistent across all methods."""
        frame = ContinueInterruptedPatternFlowStackFrame()

        # Class method
        assert frame.type() == FLOW_PATTERN_CONTINUE_INTERRUPTED

        # Serialized type
        frame_dict = frame.as_dict()
        assert frame_dict["type"] == FLOW_PATTERN_CONTINUE_INTERRUPTED

        # Deserialized type
        restored_frame = ContinueInterruptedPatternFlowStackFrame.from_dict(frame_dict)
        assert restored_frame.type() == FLOW_PATTERN_CONTINUE_INTERRUPTED

    @pytest.mark.asyncio
    async def test_continue_interrupted_yes_resumes_interrupted_agent(self):
        """Regression: continue-interrupted "Yes" transitions agent to RESUMING.

        After a KB-style digression interrupts a sub-agent and the user
        answers "Yes" to the continue-interrupted pattern, the agent frame
        must move from INTERRUPTED to RESUMING. When the agent frame is
        later popped to the top of the stack and `run_agent()` runs,
        RESUMING is what triggers attaching the
        `resumed_after_interruption=True` metadata so the sub-agent can
        pick up where it left off.

        The accompanying `AgentResumed` event is emitted later by
        `run_agent` (single canonical emitter) — not at this command-
        processing stage.
        """
        # Build a realistic stack: user flow with an INTERRUPTED agent below
        # the pattern_continue_interrupted frame the user is responding to.
        user_flow_frame = UserFlowStackFrame(
            flow_id="my_flow",
            step_id="call-research-agent",
            frame_id="user-frame-id",
            frame_type=FlowStackFrameType.REGULAR,
        )
        agent_frame = AgentStackFrame(
            frame_id="agent-frame-id",
            state=AgentState.INTERRUPTED,
            agent_id="car-research",
            flow_id="my_flow",
            step_id="call-research-agent",
            metadata={"agent_response": "What is your budget?"},
        )
        pattern_frame = ContinueInterruptedPatternFlowStackFrame(
            frame_id="pattern-frame-id",
            interrupted_flow_ids=["my_flow"],
            interrupted_flow_names=["my_flow"],
            multiple_flows_interrupted=False,
        )
        stack = DialogueStack(frames=[user_flow_frame, agent_frame, pattern_frame])
        tracker = DialogueStateTracker.from_events("test_sender", [])
        tracker.update_stack(stack)

        output_channel = MagicMock()
        nlg = MagicMock()
        domain = MagicMock()

        action = ActionContinueInterruptedFlow()
        events = await action.run(output_channel, nlg, tracker, domain)

        # `AgentResumed` is NOT emitted here. `run_agent` is the single
        # canonical emitter and surfaces the event once the agent is actually
        # re-invoked (observed via the post-resume RESUMING state below).
        assert not any(isinstance(e, AgentResumed) for e in events)

        # Apply the events to the tracker so we can inspect the resulting stack.
        # tracker.stack returns a copy each call; we need to see the post-resume
        # state of the agent frame to confirm the RESUMING transition.
        for event in events:
            tracker.update(event)
        post_resume_agent_frame = next(
            frame
            for frame in tracker.stack.frames
            if isinstance(frame, AgentStackFrame)
        )
        # Agent frame should now be in RESUMING state so that the next
        # run_agent() invocation takes the resume branch.
        assert post_resume_agent_frame.state == AgentState.RESUMING
