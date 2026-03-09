from typing import TYPE_CHECKING

from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from rasa.hooks import hookimpl
from rasa.plugin import plugin_manager
from rasa.shared.core.events import ActionExecuted

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker


async def test_after_new_user_message(default_agent: Agent) -> None:
    hook_result = []

    class TestHookPlugin:
        @hookimpl  # type: ignore[misc]
        def after_new_user_message(self, tracker: "DialogueStateTracker") -> None:
            """Triggers a tracker update notification after a new user message."""
            hook_result.append(tracker)

    plugin_manager().register(TestHookPlugin())

    message_text = "hello"
    user_message = UserMessage(message_text, sender_id="some id")

    await default_agent.handle_message(user_message)

    assert len(hook_result) == 1
    assert hook_result[0].sender_id == "some id"
    assert hook_result[0].latest_message.text == message_text


async def test_after_action_executed(agent_with_flows: Agent) -> None:
    hook_result = []

    class TestHookPlugin:
        @hookimpl  # type: ignore[misc]
        def after_action_executed(self, tracker: "DialogueStateTracker") -> None:
            """Triggers a tracker update notification after a new user message."""
            hook_result.append(tracker)

    plugin_manager().register(TestHookPlugin())

    message_text = "hello"
    user_message = UserMessage(message_text, sender_id="some id")

    await agent_with_flows.handle_message(user_message)

    assert len(hook_result) > 0

    # each tracker should have an action at the end
    for tracker in hook_result:
        assert len(tracker.events) > 0
        assert isinstance(tracker.events[-1], ActionExecuted)
