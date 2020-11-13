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
        ("data/test_nlg/responses.md", True),
        ("data/test_nlu/default_retrieval_intents.md", False),
        ("data/test_stories/stories.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", False),
    ],
)
def test_converter_filters_correct_files(training_data_file: Text, should_filter: bool):
    assert should_filter == NLGMarkdownToYamlConverter.filter(Path(training_data_file))


async def test_nlu_intents_are_converted(tmpdir: Path):
    converted_data_folder = tmpdir / "converted_data"
    os.mkdir(converted_data_folder)

    training_data_folder = tmpdir / "data"
    os.makedirs(training_data_folder, exist_ok=True)
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

    with open(training_data_file, "w") as f:
        f.write(simple_nlg_md)

    await NLGMarkdownToYamlConverter().convert_and_write(
        training_data_file, converted_data_folder
    )

    assert len(os.listdir(converted_data_folder)) == 1

    with open(f"{converted_data_folder}/responses_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "responses:\n"
            "  utter_chitchat/ask_name:\n"
            "  - text: my name is Sara, Rasa's documentation bot!\n"
            "  utter_faq/ask_location:\n"
            "  - text: We're located in the world\n"
        )
