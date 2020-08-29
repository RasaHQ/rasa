from pathlib import Path
from typing import Dict, Text

import pytest
from _pytest.tmpdir import TempdirFactory
import os

from rasa.constants import (
    DEFAULT_CORE_SUBDIRECTORY_NAME,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_E2E_TESTS_PATH,
)
from rasa.nlu.training_data.formats import RasaReader
import rasa.utils.io
from rasa import model
from rasa.core import utils
from rasa.core.domain import Domain
from rasa.importers.multi_project import MultiProjectImporter


def test_load_imports_from_directory_tree(tmp_path: Path):
    root_imports = {"imports": ["Project A"]}
    utils.dump_obj_as_yaml_to_file(tmp_path / "config.yml", root_imports)

    project_a_directory = tmp_path / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["../Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    project_b_directory = tmp_path / "Project B"
    project_b_directory.mkdir()
    project_b_imports = {"some other": ["../Project C"]}
    utils.dump_obj_as_yaml_to_file(
        project_b_directory / "config.yml", project_b_imports
    )

    project_b_subproject_directory = project_b_directory / "Project B-1"
    project_b_subproject_directory.mkdir()
    project_b_1_imports = {"imports": ["../../Project A"]}
    # Check if loading from `.yaml` also works
    utils.dump_obj_as_yaml_to_file(
        project_b_subproject_directory / "config.yaml", project_b_1_imports
    )

    # should not be imported
    subdirectory_3 = tmp_path / "Project C"
    subdirectory_3.mkdir()

    expected = [
        os.path.join(str(project_a_directory)),
        os.path.join(str(project_b_directory)),
    ]

    actual = MultiProjectImporter(str(tmp_path / "config.yml"))

    assert actual._imports == expected


def test_load_imports_without_imports(tmp_path: Path):
    empty_config = {}
    utils.dump_obj_as_yaml_to_file(tmp_path / "config.yml", empty_config)

    project_a_directory = tmp_path / "Project A"
    project_a_directory.mkdir()
    utils.dump_obj_as_yaml_to_file(project_a_directory / "config.yml", empty_config)

    project_b_directory = tmp_path / "Project B"
    project_b_directory.mkdir()
    utils.dump_obj_as_yaml_to_file(project_b_directory / "config.yml", empty_config)

    actual = MultiProjectImporter(str(tmp_path / "config.yml"))

    assert actual.is_imported(str(tmp_path / "Project C"))


@pytest.mark.parametrize("input_dict", [{}, {"imports": None}])
def test_load_from_none(input_dict: Dict, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    utils.dump_obj_as_yaml_to_file(tmp_path / "config.yml", input_dict)

    actual = MultiProjectImporter(str(config_path))

    assert actual._imports == list()


def test_load_if_subproject_is_more_specific_than_parent(tmp_path: Path,):
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(tmp_path / "config.yml", {})

    project_a_directory = tmp_path / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    actual = MultiProjectImporter(config_path)

    assert actual.is_imported(str(project_a_directory))


@pytest.mark.parametrize(
    "input_path", ["A/A/A/B", "A/A/A", "A/B/A/A", "A/A/A/B/C/D/E.type"]
)
def test_in_imports(input_path: Text, tmp_path: Path):
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(
        tmp_path / "config.yml", {"imports": ["A/A/A", "A/B/A"]}
    )

    importer = MultiProjectImporter(config_path, project_directory=os.getcwd())

    assert importer.is_imported(input_path)


@pytest.mark.parametrize("input_path", ["A/C", "A/A/B", "A/B"])
def test_not_in_imports(input_path: Text, tmp_path: Path):
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(
        tmp_path / "config.yml", {"imports": ["A/A/A", "A/B/A"]}
    )
    importer = MultiProjectImporter(config_path, project_directory=os.getcwd())

    assert not importer.is_imported(input_path)


def test_cyclic_imports(tmp_path: Path):
    project_imports = {"imports": ["Project A"]}
    utils.dump_obj_as_yaml_to_file(tmp_path / "config.yml", project_imports)

    project_a_directory = tmp_path / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["../Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    project_b_directory = tmp_path / "Project B"
    project_b_directory.mkdir()
    project_b_imports = {"imports": ["../Project A"]}
    utils.dump_obj_as_yaml_to_file(
        project_b_directory / "config.yml", project_b_imports
    )

    actual = MultiProjectImporter(str(tmp_path / "config.yml"))

    assert actual._imports == [str(project_a_directory), str(project_b_directory)]


def test_import_outside_project_directory(tmp_path: Path):
    project_imports = {"imports": ["Project A"]}
    utils.dump_obj_as_yaml_to_file(tmp_path / "config.yml", project_imports)

    project_a_directory = tmp_path / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["../Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    project_b_directory = tmp_path / "Project B"
    project_b_directory.mkdir()
    project_b_imports = {"imports": ["../Project C"]}
    utils.dump_obj_as_yaml_to_file(
        project_b_directory / "config.yml", project_b_imports
    )

    actual = MultiProjectImporter(str(project_a_directory / "config.yml"))

    assert actual._imports == [str(project_b_directory), str(tmp_path / "Project C")]


def test_importing_additional_files(tmp_path: Path):
    config = {"imports": ["bots/Bot A"]}
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(config_path, config)

    additional_file = tmp_path / "directory" / "file.md"
    additional_file.parent.mkdir()

    # create intermediate directories and fake files
    rasa.utils.io.write_text_file("""## story""", additional_file)
    selector = MultiProjectImporter(
        config_path,
        training_data_paths=[str(tmp_path / "directory"), str(additional_file)],
    )

    assert selector.is_imported(str(additional_file))
    assert str(additional_file) in selector._story_paths


def test_not_importing_not_relevant_additional_files(tmp_path: Path):
    config = {"imports": ["bots/Bot A"]}
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(config_path, config)

    additional_file = tmp_path / "directory" / "file.yml"
    additional_file.parent.mkdir()

    selector = MultiProjectImporter(
        config_path, training_data_paths=[str(tmp_path / "data"), str(additional_file)]
    )

    not_relevant_file1 = tmp_path / "data" / "another directory" / "file.yml"
    not_relevant_file1.parent.mkdir(parents=True)
    rasa.utils.io.write_text_file("", not_relevant_file1)
    not_relevant_file2 = tmp_path / "directory" / "another_file.yml"
    rasa.utils.io.write_text_file("", not_relevant_file2)

    assert not selector.is_imported(str(not_relevant_file1))
    assert not selector.is_imported(str(not_relevant_file2))


@pytest.mark.parametrize(
    "test_stories_filename,test_story",
    [
        (
            "test_stories.yml",
            """
        stories:
        - story: story test
          steps:
          - user: hello
            intent: greet
          - action: utter_greet
        """,
        ),
        (
            "conversation_tests.md",
            """
        ## story test
        * greet : "hello"
            - utter_greet
        """,
        ),
    ],
)
async def test_only_getting_e2e_conversation_tests_if_e2e_enabled(
    tmp_path: Path, test_stories_filename: Text, test_story: Text
):
    from rasa.core.training.structures import StoryGraph
    import rasa.core.training.loading as core_loading

    config = {"imports": ["bots/Bot A"]}
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(config_path, config)

    story_file = tmp_path / "bots" / "Bot A" / "data" / "stories.md"
    story_file.parent.mkdir(parents=True)
    rasa.utils.io.write_text_file(
        """
        ## story
        * greet
            - utter_greet
        """,
        story_file,
    )

    story_test_file = (
        tmp_path / "bots" / "Bot A" / DEFAULT_E2E_TESTS_PATH / test_stories_filename
    )
    story_test_file.parent.mkdir(parents=True)
    rasa.utils.io.write_text_file(test_story, story_test_file)

    selector = MultiProjectImporter(config_path)

    story_steps = await core_loading.load_data_from_resource(
        resource=str(story_test_file),
        domain=Domain.empty(),
        template_variables=None,
        use_e2e=True,
        exclusion_percentage=None,
    )

    expected = StoryGraph(story_steps)

    actual = await selector.get_stories(use_e2e=True)

    assert expected.as_story_string() == actual.as_story_string()


def test_not_importing_e2e_conversation_tests_in_project(tmp_path: Path,):
    config = {"imports": ["bots/Bot A"]}
    config_path = str(tmp_path / "config.yml")
    utils.dump_obj_as_yaml_to_file(config_path, config)

    story_file = tmp_path / "bots" / "Bot A" / "data" / "stories.md"
    story_file.parent.mkdir(parents=True)
    rasa.utils.io.write_text_file("""## story""", story_file)

    story_test_file = (
        tmp_path / "bots" / "Bot A" / DEFAULT_E2E_TESTS_PATH / "test_stories.yml"
    )
    story_test_file.parent.mkdir(parents=True)
    rasa.utils.io.write_text_file("""stories:""", story_test_file)

    selector = MultiProjectImporter(config_path)

    # Conversation tests should not be included in story paths
    assert [str(story_file)] == selector._story_paths
    assert [str(story_test_file)] == selector._e2e_story_paths


def test_single_additional_file(tmp_path: Path):
    config_path = str(tmp_path / "config.yml")
    empty_config = {}
    utils.dump_obj_as_yaml_to_file(config_path, empty_config)

    additional_file = tmp_path / "directory" / "file.yml"
    additional_file.parent.mkdir()
    rasa.utils.io.write_yaml({}, additional_file)

    selector = MultiProjectImporter(
        config_path, training_data_paths=str(additional_file)
    )

    assert selector.is_imported(str(additional_file))


async def test_multi_project_training(trained_async):
    example_directory = "data/test_multi_domain"
    config_file = os.path.join(example_directory, "config.yml")
    domain_file = os.path.join(example_directory, "domain.yml")
    files_of_root_project = os.path.join(example_directory, "data")

    trained_stack_model_path = await trained_async(
        config=config_file,
        domain=domain_file,
        training_files=files_of_root_project,
        force_training=True,
        persist_nlu_training_data=True,
    )

    unpacked = model.unpack_model(trained_stack_model_path)

    domain_file = os.path.join(
        unpacked, DEFAULT_CORE_SUBDIRECTORY_NAME, DEFAULT_DOMAIN_PATH
    )
    domain = Domain.load(domain_file)

    expected_intents = {
        "greet",
        "goodbye",
        "affirm",
        "deny",
        "mood_great",
        "mood_unhappy",
    }

    assert all([i in domain.intents for i in expected_intents])

    nlu_training_data_file = os.path.join(unpacked, "nlu", "training_data.json")
    nlu_training_data = RasaReader().read(nlu_training_data_file)

    assert expected_intents == nlu_training_data.intents

    expected_actions = [
        "utter_greet",
        "utter_cheer_up",
        "utter_did_that_help",
        "utter_happy",
        "utter_goodbye",
    ]

    assert all([a in domain.action_names for a in expected_actions])
