import os
from pathlib import Path
from typing import Text

import pytest

from rasa.nlu.training_data.converters.nlu_markdown_to_yaml_converter import (
    NLUMarkdownToYamlConverter,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


@pytest.mark.parametrize(
    "training_data_file, should_filter",
    [
        ("data/test_nlu/default_retrieval_intents.md", True),
        ("data/test_stories/stories.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", False),
    ],
)
def test_converter_filters_correct_files(training_data_file: Text, should_filter: bool):
    assert should_filter == NLUMarkdownToYamlConverter.filter(Path(training_data_file))


async def test_nlu_intents_are_converted(tmpdir: Path):
    converted_data_folder = tmpdir / "converted_data"
    os.mkdir(converted_data_folder)

    training_data_folder = tmpdir / "data/nlu"
    os.makedirs(training_data_folder, exist_ok=True)
    training_data_file = Path(training_data_folder / "nlu.md")

    simple_nlu_md = """
        ## intent:greet
        - hey
        - hello
        """

    with open(training_data_file, "w") as f:
        f.write(simple_nlu_md)

    await NLUMarkdownToYamlConverter().convert_and_write(
        training_data_file, converted_data_folder
    )

    assert len(os.listdir(converted_data_folder)) == 1

    with open(f"{converted_data_folder}/nlu_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- intent: greet\n"
            "  examples: |\n"
            "    - hey\n"
            "    - hello\n"
        )


async def test_nlu_lookup_tables_are_converted(tmpdir: Path):
    converted_data_folder = tmpdir / "converted_data"
    os.mkdir(converted_data_folder)

    training_data_folder = tmpdir / "data/nlu"
    os.makedirs(training_data_folder, exist_ok=True)
    training_data_file = Path(training_data_folder / "nlu.md")

    simple_nlu_md = f"""
    ## lookup:products.txt
      {tmpdir / "data/nlu/lookups/products.txt"}
    """

    with open(training_data_file, "w") as f:
        f.write(simple_nlu_md)

    lookup_data_folder = training_data_folder / "lookups"
    os.makedirs(lookup_data_folder, exist_ok=True)
    lookup_tables_file = lookup_data_folder / "products.txt"

    simple_lookup_table_txt = "core\n nlu\n x\n"

    with open(lookup_tables_file, "w") as f:
        f.write(simple_lookup_table_txt)

    await NLUMarkdownToYamlConverter().convert_and_write(
        training_data_file, converted_data_folder
    )

    assert len(os.listdir(converted_data_folder)) == 1

    with open(f"{converted_data_folder}/products_converted.yml", "r") as f:
        content = f.read()
        assert content == (
            f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
            "nlu:\n"
            "- lookup: products\n"
            "  examples: |\n"
            "    - core\n"
            "    - nlu\n"
            "    - x\n"
        )
