import asyncio
import os
from pathlib import Path
from typing import Text, Dict, Type, List

import pytest
from rasa.constants import DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH, DEFAULT_DATA_PATH
from rasa.core.domain import Domain
from rasa.core.events import SlotSet, UserUttered, ActionExecuted
from rasa.core.training.structures import StoryStep, StoryGraph
from rasa.importers.importer import (
    CombinedDataImporter,
    TrainingDataImporter,
    NluDataImporter,
    CoreDataImporter,
    E2EImporter,
)
from rasa.importers.rasa import RasaFileImporter

# noinspection PyUnresolvedReferences
from rasa.importers.multi_project import MultiProjectImporter

# noinspection PyUnresolvedReferences
from rasa.nlu.constants import MESSAGE_ACTION_NAME, MESSAGE_INTENT_NAME
from rasa.nlu.training_data import Message
from tests.core.conftest import project


async def test_use_of_interface():
    importer = TrainingDataImporter()

    functions_to_test = [
        lambda: importer.get_config(),
        lambda: importer.get_stories(),
        lambda: importer.get_nlu_data(),
        lambda: importer.get_domain(),
    ]
    for f in functions_to_test:
        with pytest.raises(NotImplementedError):
            await f()


async def test_combined_file_importer_with_single_importer(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)

    importer = RasaFileImporter(config_path, domain_path, [default_data_path])
    combined = CombinedDataImporter([importer])

    assert await importer.get_config() == await combined.get_config()
    actual_domain = await combined.get_domain()
    expected_domain = await importer.get_domain()
    assert hash(actual_domain) == hash(expected_domain)

    actual_training_data = await combined.get_nlu_data()
    expected_training_data = await importer.get_nlu_data()
    assert hash(actual_training_data) == hash(expected_training_data)

    expected_stories = await importer.get_stories()
    actual_stories = await combined.get_stories()

    assert actual_stories.as_story_string() == expected_stories.as_story_string()


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, [RasaFileImporter]),
        ({"importers": []}, [RasaFileImporter]),
        ({"importers": [{"name": "RasaFileImporter"}]}, [RasaFileImporter]),
        ({"importers": [{"name": "NotExistingModule"}]}, [RasaFileImporter]),
        (
            {
                "importers": [
                    {"name": "rasa.importers.multi_project.MultiProjectImporter"}
                ]
            },
            [MultiProjectImporter],
        ),
        ({"importers": [{"name": "MultiProjectImporter"}]}, [MultiProjectImporter]),
        (
            {
                "importers": [
                    {"name": "RasaFileImporter"},
                    {"name": "MultiProjectImporter"},
                ]
            },
            [RasaFileImporter, MultiProjectImporter],
        ),
    ],
)
def test_load_from_dict(
    config: Dict, expected: List[Type["TrainingDataImporter"]], project: Text
):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    actual = TrainingDataImporter.load_from_dict(
        config, config_path, domain_path, [default_data_path]
    )

    assert isinstance(actual, CombinedDataImporter)

    actual_importers = [i.__class__ for i in actual._importers]
    assert actual_importers == expected


def test_load_from_config(tmpdir: Path):
    import rasa.utils.io as io_utils

    config_path = str(tmpdir / "config.yml")

    io_utils.write_yaml_file(
        {"importers": [{"name": "MultiProjectImporter"}]}, config_path
    )

    importer = TrainingDataImporter.load_from_config(config_path)
    assert isinstance(importer, CombinedDataImporter)
    assert isinstance(importer._importers[0], MultiProjectImporter)


async def test_nlu_only(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    actual = TrainingDataImporter.load_nlu_importer_from_config(
        config_path, training_data_paths=[default_data_path]
    )

    assert isinstance(actual, NluDataImporter)

    stories = await actual.get_stories()
    assert stories.is_empty()

    domain = await actual.get_domain()
    assert domain.is_empty()

    config = await actual.get_config()
    assert config

    nlu_data = await actual.get_nlu_data()
    assert not nlu_data.is_empty()


async def test_core_only(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    actual = TrainingDataImporter.load_core_importer_from_config(
        config_path, domain_path, training_data_paths=[default_data_path]
    )

    assert isinstance(actual, CoreDataImporter)

    stories = await actual.get_stories()
    assert not stories.is_empty()

    domain = await actual.get_domain()
    assert not domain.is_empty()

    config = await actual.get_config()
    assert config

    nlu_data = await actual.get_nlu_data()
    assert nlu_data.is_empty()


async def test_import_nlu_training_data_from_e2e_stories(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    existing = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, [default_data_path]
    )

    stories = StoryGraph(
        [
            StoryStep(
                events=[
                    SlotSet("some slot", "doesn't matter"),
                    UserUttered("greet_from_stories", {"name": "greet_from_stories"}),
                    ActionExecuted("utter_greet_from_stories"),
                ]
            ),
            StoryStep(
                events=[
                    UserUttered("how are you doing?", {"name": "greet_from_stories"}),
                    ActionExecuted("utter_greet_from_stories", e2e_text="Hi Joey."),
                ]
            ),
        ]
    )

    # Patch to return our test stories
    existing.get_stories = asyncio.coroutine(lambda *args: stories)

    importer = E2EImporter(existing)

    # The wrapping `E2EImporter` simply forwards these
    assert (await existing.get_stories()).as_story_string() == (
        await importer.get_stories()
    ).as_story_string()
    assert (await existing.get_config()) == (await importer.get_config())

    # Check additional NLU training data from stories was added
    nlu_data = await importer.get_nlu_data()

    assert len(nlu_data.training_examples) > len(
        (await existing.get_nlu_data()).training_examples
    )

    expected_additional_messages = [
        Message("greet_from_stories", data={MESSAGE_INTENT_NAME: "greet_from_stories"}),
        Message("", data={MESSAGE_ACTION_NAME: "utter_greet_from_stories"}),
        Message("how are you doing?", data={MESSAGE_INTENT_NAME: "greet_from_stories"}),
        Message("Hi Joey.", data={MESSAGE_ACTION_NAME: "utter_greet_from_stories"}),
    ]

    assert all(m in nlu_data.training_examples for m in expected_additional_messages)


async def test_import_nlu_training_data_with_default_actions(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    existing = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, [default_data_path]
    )

    importer = E2EImporter(existing)

    # Check additional NLU training data from domain was added
    nlu_data = await importer.get_nlu_data()

    assert len(nlu_data.training_examples) > len(
        (await existing.get_nlu_data()).training_examples
    )

    from rasa.core.actions import action

    extended_training_data = await importer.get_nlu_data()
    assert all(
        Message("", data={MESSAGE_ACTION_NAME: action_name})
        in extended_training_data.training_examples
        for action_name in action.default_action_names()
    )


async def test_adding_e2e_actions_to_domain(project: Text):
    config_path = os.path.join(project, DEFAULT_CONFIG_PATH)
    domain_path = os.path.join(project, DEFAULT_DOMAIN_PATH)
    default_data_path = os.path.join(project, DEFAULT_DATA_PATH)
    existing = TrainingDataImporter.load_from_dict(
        {}, config_path, domain_path, [default_data_path]
    )

    additional_actions = ["Hi Joey.", "it's sunny outside."]
    stories = StoryGraph(
        [
            StoryStep(
                events=[
                    UserUttered("greet_from_stories", {"name": "greet_from_stories"}),
                    ActionExecuted("utter_greet_from_stories"),
                ]
            ),
            StoryStep(
                events=[
                    UserUttered("how are you doing?", {"name": "greet_from_stories"}),
                    ActionExecuted(
                        additional_actions[0], e2e_text=additional_actions[0]
                    ),
                    ActionExecuted(
                        additional_actions[1], e2e_text=additional_actions[1]
                    ),
                ]
            ),
        ]
    )

    # Patch to return our test stories
    existing.get_stories = asyncio.coroutine(lambda *args: stories)

    importer = E2EImporter(existing)

    domain = await importer.get_domain()

    assert all(action_name in domain.action_names for action_name in additional_actions)
