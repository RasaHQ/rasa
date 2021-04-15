from rasa.core.agent import Agent
from rasa.shared.core.training_data.story_writer.markdown_story_writer import (
    MarkdownStoryWriter,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.events import (
    ActionExecuted,
    UserUttered,
)
from rasa.shared.core.constants import (
    ACTION_UNLIKELY_INTENT_NAME,
)


async def test_tracker_dump_e2e_story(default_agent: Agent):
    sender_id = "test_tracker_dump_e2e_story"

    await default_agent.handle_text("/greet", sender_id=sender_id)
    await default_agent.handle_text("/goodbye", sender_id=sender_id)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)

    story = tracker.export_stories(MarkdownStoryWriter(), e2e=True)
    assert story.strip().split("\n") == [
        "## test_tracker_dump_e2e_story",
        "* greet: /greet",
        "    - utter_greet",
        "* goodbye: /goodbye",
        "    - utter_goodbye",
    ]


def test_markdown_writer_doesnt_dump_action_unlikely_intent():
    events = [
        ActionExecuted("utter_hello"),
        ActionExecuted(ACTION_UNLIKELY_INTENT_NAME, metadata={"key1": "value1"}),
        ActionExecuted("utter_bye"),
    ]
    tracker = DialogueStateTracker.from_events("default", events)
    story = tracker.export_stories(MarkdownStoryWriter(), e2e=True)
    assert story.strip().split("\n") == [
        "## default",
        "    - utter_hello",
        "    - utter_bye",
    ]