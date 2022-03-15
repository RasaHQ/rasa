import os
from pathlib import Path
from typing import Text

import pytest
from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader

from rasa.nlu.training_data.converters.nlu_markdown_to_yaml_converter import (
    NLUToYamlConverter,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


@pytest.mark.parametrize(
    "training_data_file, should_filter",
    [
        ("data/test_md/default_retrieval_intents.md", True),
        ("data/test_stories/stories.md", False),
        ("data/test_yaml_stories/rules_without_stories.yml", False),
        ("data/examples/rasa/demo-rasa.json", True),
    ],
)
def test_converter_filters_correct_files(training_data_file: Text, should_filter: bool):
    assert should_filter == NLUToYamlConverter.filter(Path(training_data_file))


async def test_nlu_intents_are_converted(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    training_data_folder = tmp_path / "data" / "nlu"
    training_data_folder.mkdir(parents=True)
    training_data_file = Path(training_data_folder / "nlu.md")

    simple_nlu_md = """
        ## intent:greet
        - hey
        - hello
        """

    training_data_file.write_text(simple_nlu_md)

    with pytest.warns(None) as warnings:
        await NLUToYamlConverter().convert_and_write(
            training_data_file, converted_data_folder
        )
    assert not warnings

    assert len(os.listdir(converted_data_folder)) == 1

    converted_file = converted_data_folder / "nlu_converted.yml"
    content = converted_file.read_text()

    assert content == (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        "nlu:\n"
        "- intent: greet\n"
        "  examples: |\n"
        "    - hey\n"
        "    - hello\n"
    )


async def test_nlu_lookup_tables_are_converted(tmp_path: Path):
    converted_data_folder = tmp_path / "converted_data"
    converted_data_folder.mkdir()

    training_data_folder = tmp_path / "data" / "nlu"
    training_data_folder.mkdir(parents=True)
    training_data_file = Path(training_data_folder / "nlu.md")

    lookup_data_folder = training_data_folder / "lookups"
    lookup_data_folder.mkdir()
    lookup_tables_file = lookup_data_folder / "products.txt"

    simple_lookup_table_txt = "core\n nlu\n x\n"

    lookup_tables_file.write_text(simple_lookup_table_txt)

    simple_nlu_md = f"""
    ## lookup:products.txt
      {lookup_tables_file}
    """

    training_data_file.write_text(simple_nlu_md)

    with pytest.warns(None) as warnings:
        await NLUToYamlConverter().convert_and_write(
            training_data_file, converted_data_folder
        )
    assert not warnings

    assert len(os.listdir(converted_data_folder)) == 1

    converted_file = converted_data_folder / "products_converted.yml"
    content = converted_file.read_text()
    assert content == (
        f'version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"\n'
        "nlu:\n"
        "- lookup: products\n"
        "  examples: |\n"
        "    - core\n"
        "    - nlu\n"
        "    - x\n"
    )


async def test_convert_json_to_yaml(tmp_path: Path):
    test_data_dir = Path("data", "examples", "rasa")

    with pytest.warns(None) as warnings:
        await NLUToYamlConverter().convert_and_write(
            Path("data/examples/rasa/demo-rasa.json"), tmp_path
        )
    assert not warnings

    assert len(os.listdir(tmp_path)) == 1

    converted_file = tmp_path / "demo-rasa_converted.yml"

    actual_data = RasaYAMLReader().read(converted_file)
    expected = RasaYAMLReader().read(test_data_dir / "demo-rasa.yml")

    assert len(actual_data.training_examples) == len(expected.training_examples)
    assert len(actual_data.entity_synonyms) == len(expected.entity_synonyms)
    assert len(actual_data.regex_features) == len(expected.regex_features)
    assert len(actual_data.lookup_tables) == len(expected.lookup_tables)
    assert actual_data.entities == expected.entities
