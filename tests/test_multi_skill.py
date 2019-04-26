from pathlib import Path

import pytest
from _pytest.tmpdir import TempdirFactory
import os

from rasa.core import utils
from rasa.multi_skill import SkillSelector


def test_load_imports_from_directory_tree(tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    root_imports = {"imports": ["Skill A"]}
    utils.dump_obj_as_yaml_to_file(root / "config.yml", root_imports)

    skill_a_directory = root / "Skill A"
    skill_a_directory.mkdir()
    skill_a_imports = {"imports": ["Skill B"]}
    utils.dump_obj_as_yaml_to_file(skill_a_directory / "config.yml", skill_a_imports)

    skill_b_directory = root / "Skill B"
    skill_b_directory.mkdir()
    skill_b_imports = {"some other": ["Skill C"]}
    utils.dump_obj_as_yaml_to_file(skill_b_directory / "config.yml", skill_b_imports)

    skill_b_subskill_directory = skill_b_directory / "Skill B-1"
    skill_b_subskill_directory.mkdir()
    skill_b_1_imports = {"imports": ["Skill A"]}
    # Check if loading from `.yaml` also works
    utils.dump_obj_as_yaml_to_file(
        skill_b_subskill_directory / "config.yaml", skill_b_1_imports
    )

    # should not be imported
    subdirectory_3 = root / "Skill C"
    subdirectory_3.mkdir()

    actual = SkillSelector.load(root / "config.yml", [root])
    expected = {
        os.path.join(str(skill_a_directory)),
        os.path.join(str(skill_b_directory)),
    }

    assert actual.imports == expected


def test_load_imports_without_imports(tmpdir_factory: TempdirFactory):
    empty_config = {}
    root = tmpdir_factory.mktemp("Parent Bot")
    utils.dump_obj_as_yaml_to_file(root / "config.yml", empty_config)

    skill_a_directory = root / "Skill A"
    skill_a_directory.mkdir()
    utils.dump_obj_as_yaml_to_file(skill_a_directory / "config.yml", empty_config)

    skill_b_directory = root / "Skill B"
    skill_b_directory.mkdir()
    utils.dump_obj_as_yaml_to_file(skill_b_directory / "config.yml", empty_config)

    actual = SkillSelector.load(root / "config.yml", [root])

    assert actual.imports == {str(root)}

    assert actual.is_imported(root / "Skill C")


@pytest.mark.parametrize("input_dict", [{}, {"imports": None}])
def test_load_from_none(input_dict):
    actual = SkillSelector._from_dict(input_dict, Path("."))

    assert actual.imports == set()


def test_load_if_skill_paths_are_files(tmpdir):
    empty_config = {}
    config_path = tmpdir / "config.yml"
    utils.dump_obj_as_yaml_to_file(config_path, empty_config)

    other_file = tmpdir / "nlu.md"
    utils.dump_obj_as_yaml_to_file(other_file, ["list", "of", "names"])

    actual = SkillSelector.load(config_path, other_file)

    assert actual.is_imported(other_file)


def test_load_if_subskill_is_more_specific_than_parent(tmpdir_factory: TempdirFactory):
    root = tmpdir_factory.mktemp("Parent Bot")
    config_path = root / "config.yml"
    utils.dump_obj_as_yaml_to_file(root / "config.yml", {})

    skill_a_directory = root / "Skill A"
    skill_a_directory.mkdir()
    skill_a_imports = {"imports": ["Skill B"]}
    utils.dump_obj_as_yaml_to_file(skill_a_directory / "config.yml", skill_a_imports)

    actual = SkillSelector.load(config_path, [root])

    assert actual.is_imported(skill_a_directory)


@pytest.mark.parametrize(
    "input_path", ["A/A/A/B", "A/A/A", "A/B/A/A", "A/A/A/B/C/D/E.type"]
)
def test_in_imports(input_path):
    importer = SkillSelector({"A/A/A", "A/B/A"})

    assert importer.is_imported(input_path)


@pytest.mark.parametrize("input_path", ["A/C", "A/A/B", "A/B"])
def test_not_in_imports(input_path):
    importer = SkillSelector({"A/A/A", "A/B/A"})

    assert not importer.is_imported(input_path)
