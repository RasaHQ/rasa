import os
from pathlib import Path
from typing import Text, Dict, Type, List

import pytest
from rasa.constants import DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH, DEFAULT_DATA_PATH
from rasa.core.domain import Domain
from rasa.importers.importer import CombinedFileImporter, TrainingFileImporter
from rasa.importers.simple import SimpleFileImporter

# noinspection PyUnresolvedReferences
from rasa.importers.skill import SkillSelector
from tests.core.conftest import project


async def test_use_of_interface():
    importer = TrainingFileImporter()

    functions_to_test = [
        lambda: importer.get_config(),
        lambda: importer.get_story_data(),
        lambda: importer.get_nlu_data(),
        lambda: importer.get_domain(),
    ]
    for f in functions_to_test:
        with pytest.raises(NotImplementedError):
            await f()


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


async def test_simple_file_importer_with_invalid_domain(tmp_path: Path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("")
    importer = TrainingFileImporter.load_from_dict({}, str(config_file), None, [])

    actual = await importer.get_domain()
    assert actual.as_dict() == Domain.empty().as_dict()


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
        ({}, [SimpleFileImporter]),
        ({"importers": []}, [SimpleFileImporter]),
        ({"importers": [{"name": "SimpleFileImporter"}]}, [SimpleFileImporter]),
        ({"importers": [{"name": "SkillSelector"}]}, [SkillSelector]),
        (
            {"importers": [{"name": "SimpleFileImporter"}, {"name": "SkillSelector"}]},
            [SimpleFileImporter, SkillSelector],
        ),
    ],
)
def test_load_from_dict(
    config: Dict, expected: List[Type["TrainingFileImporter"]], project: Text
):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    actual = TrainingFileImporter.load_from_dict(
        config, config_path, domain_path, [default_data_path]
    )

    assert isinstance(actual, CombinedFileImporter)

    actual_importers = [i.__class__ for i in actual._importers]
    assert actual_importers == expected
