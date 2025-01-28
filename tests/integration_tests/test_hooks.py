from pathlib import Path
from typing import TYPE_CHECKING

from rasa.anonymization.anonymization_pipeline import (
    BackgroundAnonymizationPipeline,
    SyncAnonymizationPipeline,
)
from rasa.core.agent import Agent
from rasa.core.channels.channel import UserMessage
from rasa.hooks import hookimpl
from rasa.plugin import plugin_manager
from rasa.shared.core.events import ActionExecuted

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker


def test_get_anonymization_pipeline_no_endpoints() -> None:
    plugin_manager().hook.init_anonymization_pipeline(endpoints_file=None)
    pipeline = plugin_manager().hook.get_anonymization_pipeline()
    assert pipeline is None


def test_get_anonymization_pipeline() -> None:
    endpoints_file = (
        Path(__file__).parent.parent.parent / "data" / "anonymization" / "endpoints.yml"
    )

    # first load the anonymization pipeline
    plugin_manager().hook.init_anonymization_pipeline(endpoints_file=endpoints_file)

    pipeline = plugin_manager().hook.get_anonymization_pipeline()

    assert isinstance(pipeline, BackgroundAnonymizationPipeline)
    assert isinstance(pipeline.anonymization_pipeline, SyncAnonymizationPipeline)
    assert len(pipeline.anonymization_pipeline.orchestrators) == 2

    pipeline.stop()


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


async def test_after_action_executed(default_agent: Agent) -> None:
    hook_result = []

    class TestHookPlugin:
        @hookimpl  # type: ignore[misc]
        def after_action_executed(self, tracker: "DialogueStateTracker") -> None:
            """Triggers a tracker update notification after a new user message."""
            hook_result.append(tracker)

    plugin_manager().register(TestHookPlugin())

    message_text = "hello"
    user_message = UserMessage(message_text, sender_id="some id")

    await default_agent.handle_message(user_message)

    assert len(hook_result) > 0

    # each tracker should have an action at the end
    for tracker in hook_result:
        assert len(tracker.events) > 0
        assert isinstance(tracker.events[-1], ActionExecuted)
