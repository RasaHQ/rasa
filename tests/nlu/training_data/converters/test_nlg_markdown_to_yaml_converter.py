import os
from pathlib import Path
from typing import Text

import pytest

from rasa.nlu.training_data.converters.nlg_markdown_to_yaml_converter import (
    NLGMarkdownToYamlConverter,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


@pytest.mark.parametrize(
    "training_data_file, should_filter",
    [
        ("data/test_md/responses.md", True),
        ("data/test_md/default_retrieval_intents.md", False),
        ("data/test_stories/stories.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", False),
    ],
)
def test_converter_filters_correct_files(training_data_file: Text, should_filter: bool):
    assert should_filter == NLGMarkdownToYamlConverter.filter(Path(training_data_file))


async def test_nlu_intents_are_converted(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    training_data_folder = tmp_path / "data"
    training_data_folder.mkdir()
    training_data_file = Path(training_data_folder) / "responses.md"

    simple_nlg_md = (
        # missing utter_
        "## ask name\n"
        "* chitchat/ask_name\n"
        "- my name is Sara, Rasa's documentation bot!\n\n"
        # utter_ is already here
        "## ask location\n"
        "* utter_faq/ask_location\n"
        "- We're located in the world\n\n"
    )

    training_data_file.write_text(simple_nlg_md)

    with pytest.warns(None) as warnings:
        await NLGMarkdownToYamlConverter().convert_and_write(
            training_data_file, converted_data_folder
        )

    assert not warnings

    assert len(os.listdir(converted_data_folder)) == 1

    converted_responses = converted_data_folder / "responses_converted.yml"
    content = converted_responses.read_text()
    assert content == (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        "responses:\n"
        "  utter_chitchat/ask_name:\n"
        "  - text: my name is Sara, Rasa's documentation bot!\n"
        "  utter_faq/ask_location:\n"
        "  - text: We're located in the world\n"
    )
