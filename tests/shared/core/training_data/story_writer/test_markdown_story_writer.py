import pytest

from rasa.core.agent import Agent
from rasa.shared.core.training_data.story_writer.markdown_story_writer import (
    MarkdownStoryWriter,
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
