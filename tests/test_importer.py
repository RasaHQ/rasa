import os
from typing import Text, Dict, Type

import pytest
from rasa.constants import DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH, DEFAULT_DATA_PATH
from rasa.importers.importer import (
    SimpleFileImporter,
    CombinedFileImporter,
    TrainingFileImporter,
)

# noinspection PyUnresolvedReferences
from tests.core.conftest import project


async def test_simple_file_importer(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    importer = SimpleFileImporter(config_path, domain_path, [default_data_path])

    domain = await importer.get_domain()
    assert len(domain.intents) == 6
    assert domain.slots == []
    assert domain.entities == []
    assert len(domain.action_names) == 13
    assert len(domain.templates) == 5

    stories = await importer.get_story_data()
    assert len(stories.story_steps) == 4

    nlu_data = await importer.get_nlu_data("en")
    assert len(nlu_data.intents) == 6
    assert len(nlu_data.intent_examples) == 39


async def test_combined_file_importer_with_single_importer(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    importer = SimpleFileImporter(config_path, domain_path, [default_data_path])
    combined = CombinedFileImporter([importer])

    assert await importer.get_config() == await combined.get_config()
    assert (await importer.get_domain()).as_dict() == (
        await combined.get_domain()
    ).as_dict()
    assert (await importer.get_nlu_data()).as_json() == (
        await combined.get_nlu_data()
    ).as_json()

    expected_stories = await importer.get_story_data()
    actual_stories = await combined.get_story_data()

    assert actual_stories.as_story_string() == expected_stories.as_story_string()


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, SimpleFileImporter),
        ({"importers": []}, SimpleFileImporter),
        ({"importers": [{"name": "SimpleFileImporter"}]}, SimpleFileImporter),
    ],
)
def test_load_from_dict(
    config: Dict, expected: Type["TrainingFileImporter"], project: Text
):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    print (config)
    actual = TrainingFileImporter.load_from_dict(
        config, config_path, domain_path, [default_data_path]
    )

    assert isinstance(actual, CombinedFileImporter)
    assert len(actual._importers) == 1
    assert isinstance(actual._importers[0], expected)
