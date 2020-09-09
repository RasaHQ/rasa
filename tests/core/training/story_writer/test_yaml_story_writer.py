from pathlib import Path
import textwrap
from typing import Text

import pytest

from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, UserUttered
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.story_reader.markdown_story_reader import MarkdownStoryReader
from rasa.core.training.story_reader.yaml_story_reader import YAMLStoryReader
from rasa.core.training.story_writer.yaml_story_writer import YAMLStoryWriter


@pytest.mark.parametrize(
    "input_md_file, input_yaml_file",
    [
        ["data/test_stories/stories.md", "data/test_yaml_stories/stories.yml"],
        [
            "data/test_stories/stories_defaultdomain.md",
            "data/test_yaml_stories/stories_defaultdomain.yml",
        ],
    ],
)
async def test_simple_story(
    tmpdir: Path, default_domain: Domain, input_md_file: Text, input_yaml_file: Text
):

    original_md_reader = MarkdownStoryReader(
        default_domain, None, False, input_yaml_file, unfold_or_utterances=False,
    )
    original_md_story_steps = await original_md_reader.read_from_file(input_md_file)

    assert not YAMLStoryWriter.stories_contain_loops(original_md_story_steps)

    original_yaml_reader = YAMLStoryReader(default_domain, None, False)
    original_yaml_story_steps = await original_yaml_reader.read_from_file(
        input_yaml_file
    )

    target_story_filename = tmpdir / "test.yml"
    writer = YAMLStoryWriter()
    writer.dump(target_story_filename, original_md_story_steps)

    processed_yaml_reader = YAMLStoryReader(default_domain, None, False)
    processed_yaml_story_steps = await processed_yaml_reader.read_from_file(
        target_story_filename
    )

    assert len(processed_yaml_story_steps) == len(original_yaml_story_steps)
    for processed_step, original_step in zip(
        processed_yaml_story_steps, original_yaml_story_steps
    ):
        assert len(processed_step.events) == len(original_step.events)


async def test_forms_are_converted(default_domain: Domain):
    original_md_reader = MarkdownStoryReader(
        default_domain, None, False, unfold_or_utterances=False,
    )
    original_md_story_steps = await original_md_reader.read_from_file(
        "data/test_stories/stories_form.md"
    )

    assert YAMLStoryWriter.stories_contain_loops(original_md_story_steps)

    writer = YAMLStoryWriter()

    with pytest.warns(None) as record:
        writer.dumps(original_md_story_steps)

    assert len(record) == 0


def test_yaml_writer_dumps_user_messages():
    events = [
        UserUttered("Hello", {"name": "greet"}),
        ActionExecuted("utter_greet"),
    ]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps)

    assert (
        dump.strip()
        == textwrap.dedent(
            """
        version: "2.0"
        stories:
        - story: default
          steps:
          - intent: greet
            user: |-
              Hello
          - action: utter_greet

    """
        ).strip()
    )


def test_yaml_writer_avoids_dumping_not_existing_user_messages():
    events = [
        UserUttered("greet", {"name": "greet"}),
        ActionExecuted("utter_greet"),
    ]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps)

    assert (
        dump.strip()
        == textwrap.dedent(
            """
        version: "2.0"
        stories:
        - story: default
          steps:
          - intent: greet
          - action: utter_greet

    """
        ).strip()
    )
