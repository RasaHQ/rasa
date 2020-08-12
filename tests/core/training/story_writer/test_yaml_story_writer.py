from pathlib import Path
from typing import Text

import pytest

from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter
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
        RegexInterpreter(),
        default_domain,
        None,
        False,
        input_yaml_file,
        unfold_or_utterances=False,
    )
    original_md_story_steps = await original_md_reader.read_from_file(input_md_file)

    original_yaml_reader = YAMLStoryReader(
        RegexInterpreter(), default_domain, None, False
    )
    original_yaml_story_steps = await original_yaml_reader.read_from_file(
        input_yaml_file
    )

    target_story_filename = tmpdir / "test.yml"
    writer = YAMLStoryWriter()
    writer.dump(target_story_filename, original_md_story_steps)

    processed_yaml_reader = YAMLStoryReader(
        RegexInterpreter(), default_domain, None, False
    )
    processed_yaml_story_steps = await processed_yaml_reader.read_from_file(
        target_story_filename
    )

    assert len(processed_yaml_story_steps) == len(original_yaml_story_steps)
    for processed_step, original_step in zip(
        processed_yaml_story_steps, original_yaml_story_steps
    ):
        assert len(processed_step.events) == len(original_step.events)


async def test_forms_are_skipped_with_warning(default_domain: Domain):
    original_md_reader = MarkdownStoryReader(
        RegexInterpreter(), default_domain, None, False, unfold_or_utterances=False,
    )
    original_md_story_steps = await original_md_reader.read_from_file(
        "data/test_stories/stories_form.md"
    )

    writer = YAMLStoryWriter()

    with pytest.warns(UserWarning) as record:
        writer.dumps(original_md_story_steps)

    # We skip 5 stories with the forms and warn users
    assert len(record) == 5
