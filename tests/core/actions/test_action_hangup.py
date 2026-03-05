from unittest.mock import AsyncMock, MagicMock

from rasa.core.actions.action_hangup import ActionHangup
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.session_end import FLOW_PATTERN_SESSION_END
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SessionEnded
from rasa.shared.core.trackers import DialogueStateTracker


def _make_tracker(flow_id: str) -> DialogueStateTracker:
    """Create a tracker with the given flow_id as the active flow."""
    tracker = DialogueStateTracker.from_events("test_sender", evts=[])
    stack = DialogueStack.from_dict(
        [
            {
                "frame_id": "TESTFRAME1",
                "flow_id": flow_id,
                "step_id": "start",
                "type": "flow",
                "frame_type": "regular",
            }
        ]
    )
    tracker.update_stack(stack)
    return tracker


async def test_action_hangup_user_disconnected() -> None:
    """When active flow is pattern_session_end, return SessionEnded with
    'user disconnected' reason without calling channel.hangup."""
    tracker = _make_tracker(FLOW_PATTERN_SESSION_END)

    channel = MagicMock(spec=CollectingOutputChannel)
    channel.hangup = AsyncMock()

    action = ActionHangup()
    events = await action.run(
        channel, NaturalLanguageGenerator(), tracker, Domain.empty()
    )

    assert len(events) == 1
    assert isinstance(events[0], SessionEnded)
    assert events[0].metadata == {"_reason": "user disconnected"}
    channel.hangup.assert_not_called()


async def test_action_hangup_bot_disconnected() -> None:
    """When active flow is not pattern_session_end, call channel.hangup and
    return SessionEnded with 'bot disconnected' reason."""
    tracker = _make_tracker("some_other_flow")

    channel = MagicMock(spec=CollectingOutputChannel)
    channel.hangup = AsyncMock()

    action = ActionHangup()
    events = await action.run(
        channel, NaturalLanguageGenerator(), tracker, Domain.empty()
    )

    assert len(events) == 1
    assert isinstance(events[0], SessionEnded)
    assert events[0].metadata == {"_reason": "bot disconnected"}
    channel.hangup.assert_called_once_with("test_sender")
