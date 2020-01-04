from typing import Dict, Text

import pytest
from _pytest.tmpdir import TempdirFactory
import os

from rasa.constants import DEFAULT_CORE_SUBDIRECTORY_NAME, DEFAULT_DOMAIN_PATH
from rasa.nlu.training_data.formats import RasaReader
from rasa import model
from rasa.core import utils
from rasa.core.domain import Domain
from rasa.importers.multi_project import MultiProjectImporter


def test_load_imports_from_directory_tree(tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    root_imports = {"imports": ["Project A"]}
    utils.dump_obj_as_yaml_to_file(root / "config.yml", root_imports)

    project_a_directory = root / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["../Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    project_b_directory = root / "Project B"
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
    subdirectory_3 = root / "Project C"
    subdirectory_3.mkdir()

    expected = [
        os.path.join(str(project_a_directory)),
        os.path.join(str(project_b_directory)),
    ]

    actual = MultiProjectImporter(str(root / "config.yml"))

    assert actual._imports == expected


def test_load_imports_without_imports(tmpdir_factory: TempdirFactory):
    empty_config = {}
    root = tmpdir_factory.mktemp("Parent Bot")
    utils.dump_obj_as_yaml_to_file(root / "config.yml", empty_config)

    project_a_directory = root / "Project A"
    project_a_directory.mkdir()
    utils.dump_obj_as_yaml_to_file(project_a_directory / "config.yml", empty_config)

    project_b_directory = root / "Project B"
    project_b_directory.mkdir()
    utils.dump_obj_as_yaml_to_file(project_b_directory / "config.yml", empty_config)

    actual = MultiProjectImporter(str(root / "config.yml"))

    assert actual.is_imported(str(root / "Project C"))


@pytest.mark.parametrize("input_dict", [{}, {"imports": None}])
def test_load_from_none(input_dict: Dict, tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config_path = root / "config.yml"
    utils.dump_obj_as_yaml_to_file(root / "config.yml", input_dict)

    actual = MultiProjectImporter(str(config_path))

    assert actual._imports == list()


def test_load_if_subproject_is_more_specific_than_parent(
    tmpdir_factory: TempdirFactory,
):
    root = tmpdir_factory.mktemp("Parent Bot")
    config_path = str(root / "config.yml")
    utils.dump_obj_as_yaml_to_file(root / "config.yml", {})

    project_a_directory = root / "Project A"
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
def test_in_imports(input_path: Text, tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config_path = str(root / "config.yml")
    utils.dump_obj_as_yaml_to_file(root / "config.yml", {"imports": ["A/A/A", "A/B/A"]})

    importer = MultiProjectImporter(config_path, project_directory=os.getcwd())

    assert importer.is_imported(input_path)


@pytest.mark.parametrize("input_path", ["A/C", "A/A/B", "A/B"])
def test_not_in_imports(input_path: Text, tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config_path = str(root / "config.yml")
    utils.dump_obj_as_yaml_to_file(root / "config.yml", {"imports": ["A/A/A", "A/B/A"]})
    importer = MultiProjectImporter(config_path, project_directory=os.getcwd())

    assert not importer.is_imported(input_path)


def test_cyclic_imports(tmpdir_factory):
    root = tmpdir_factory.mktemp("Parent Bot")
    project_imports = {"imports": ["Project A"]}
    utils.dump_obj_as_yaml_to_file(root / "config.yml", project_imports)

    project_a_directory = root / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["../Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    project_b_directory = root / "Project B"
    project_b_directory.mkdir()
    project_b_imports = {"imports": ["../Project A"]}
    utils.dump_obj_as_yaml_to_file(
        project_b_directory / "config.yml", project_b_imports
    )

    actual = MultiProjectImporter(str(root / "config.yml"))

    assert actual._imports == [str(project_a_directory), str(project_b_directory)]


def test_import_outside_project_directory(tmpdir_factory):
    root = tmpdir_factory.mktemp("Parent Bot")
    project_imports = {"imports": ["Project A"]}
    utils.dump_obj_as_yaml_to_file(root / "config.yml", project_imports)

    project_a_directory = root / "Project A"
    project_a_directory.mkdir()
    project_a_imports = {"imports": ["../Project B"]}
    utils.dump_obj_as_yaml_to_file(
        project_a_directory / "config.yml", project_a_imports
    )

    project_b_directory = root / "Project B"
    project_b_directory.mkdir()
    project_b_imports = {"imports": ["../Project C"]}
    utils.dump_obj_as_yaml_to_file(
        project_b_directory / "config.yml", project_b_imports
    )

    actual = MultiProjectImporter(str(project_a_directory / "config.yml"))

    assert actual._imports == [str(project_b_directory), str(root / "Project C")]


def test_importing_additional_files(tmpdir_factory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config = {"imports": ["bots/Bot A"]}
    config_path = str(root / "config.yml")
    utils.dump_obj_as_yaml_to_file(config_path, config)

    additional_file = root / "directory" / "file.md"

    # create intermediate directories and fake files
    additional_file.write("""## story""", ensure=True)
    selector = MultiProjectImporter(
        config_path, training_data_paths=[str(root / "directory"), str(additional_file)]
    )

    assert selector.is_imported(str(additional_file))
    assert str(additional_file) in selector._story_paths


def test_not_importing_not_relevant_additional_files(tmpdir_factory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config = {"imports": ["bots/Bot A"]}
    config_path = str(root / "config.yml")
    utils.dump_obj_as_yaml_to_file(config_path, config)

    additional_file = root / "directory" / "file.yml"
    selector = MultiProjectImporter(
        config_path, training_data_paths=[str(root / "data"), str(additional_file)]
    )

    not_relevant_file1 = root / "data" / "another directory" / "file.yml"
    not_relevant_file1.write({}, ensure=True)
    not_relevant_file2 = root / "directory" / "another_file.yml"
    not_relevant_file2.write({}, ensure=True)

    assert not selector.is_imported(str(not_relevant_file1))
    assert not selector.is_imported(str(not_relevant_file2))


def test_single_additional_file(tmpdir_factory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config_path = str(root / "config.yml")
    empty_config = {}
    utils.dump_obj_as_yaml_to_file(config_path, empty_config)

    additional_file = root / "directory" / "file.yml"
    additional_file.write({}, ensure=True)

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
