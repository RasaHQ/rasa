import os
from pathlib import Path
from typing import Text
import pytest

from rasa.core.training.converters.story_markdown_to_yaml_converter import (
    StoryMarkdownToYamlConverter,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


@pytest.mark.parametrize(
    "training_data_file, should_filter",
    [
        ("data/test_stories/stories.md", True),
        ("data/test_nlu/default_retrieval_intents.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", False),
    ],
)
def test_converter_filters_correct_files(training_data_file: Text, should_filter: bool):
    assert should_filter == StoryMarkdownToYamlConverter.filter(
        Path(training_data_file)
    )


async def test_stories_are_converted(tmpdir: Path):
    converted_data_folder = tmpdir / "converted_data"
    os.mkdir(converted_data_folder)

    training_data_folder = tmpdir / "data/core"
    os.makedirs(training_data_folder, exist_ok=True)
    training_data_file = Path(training_data_folder / "stories.md")

    simple_story_md = """
    ## happy path
    * greet OR goodbye
        - utter_greet
        - form{"name": null}
        - slot{"name": ["value1", "value2"]}
    """

    with open(training_data_file, "w") as f:
        f.write(simple_story_md)

    await StoryMarkdownToYamlConverter().convert_and_write(
        training_data_file, converted_data_folder
    )

    assert len(os.listdir(converted_data_folder)) == 1

    with open(f"{converted_data_folder}/stories_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "stories:\n"
            "- story: happy path\n"
            "  steps:\n"
            "  - or:\n"
            "    - intent: greet\n"
            "    - intent: goodbye\n"
            "  - action: utter_greet\n"
            "  - active_loop: null\n"
            "  - slot_was_set:\n"
            "    - name:\n"
            "      - value1\n"
            "      - value2\n"
        )
