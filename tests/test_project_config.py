import pytest
import shutil
from pathlib import Path
from _pytest.monkeypatch import MonkeyPatch
from rasa.project_config import ProjectConfig


@pytest.fixture(scope="module")
def test_project(tmpdir: Path):
    pass


@pytest.mark.skip(reason="wait until scaffold is merged")
def test_default_project_file(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Test default file in scaffold dir works."""
    shutil.copyfile(
        "rasa/cli/initial_project/project.yml",
        tmp_path,
    )
    project_config = ProjectConfig()
    monkeypatch.chdir(tmp_path)
    assert project_config["nlu"] == "data/nlu.yml"


def test_short_project_file(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Test short project file works."""
    shutil.copyfile(
        "data/test_project_config/short_project_config.yml",
        tmp_path / "project.yml",
    )
    monkeypatch.chdir(tmp_path)
    project_config = ProjectConfig()
    # test custom paths
    assert project_config["nlu"] == ["data/my_nlu_file.yml"]
    assert project_config["domain"] == ["top_level_domain.yml"]
    assert project_config["models"] == "custom_models/"
    assert project_config["actions"] == "actions_dir/"
    assert project_config["importers"] == [
        {
            "name": "module.CustomImporter",
            "parameter1": "value1",
            "parameter2": "value2",
        },
        {
            "name": "RasaFileImporter",
        },
    ]
    # test default paths
    assert project_config["rules"] == ["data/rules.yml"]
    assert project_config["stories"] == ["data/stories.yml"]
    assert project_config["config"] == "config.yml"
    assert project_config["test_data"] == ["tests/"]
    assert project_config["train_test_split"] == "train_test_split/"
    assert project_config["results"] == "results/"


def test_no_project_file_no_config_file(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Test defaults are in place if both project and config are missing."""
    monkeypatch.chdir(tmp_path)
    project_config = ProjectConfig()

    assert project_config["nlu"] == ["data/my_nlu_file.yml"]
    assert project_config["domain"] == ["top_level_domain.yml"]
    assert project_config["models"] == "custom_models/"
    assert project_config["actions"] == "actions_dir/"
    assert project_config["importers"] == [
        {
            "name": "module.CustomImporter",
            "parameter1": "value1",
            "parameter2": "value2",
        },
        {
            "name": "RasaFileImporter",
        },
    ]
    # test default paths
    assert project_config["rules"] == ["data/rules.yml"]
    assert project_config["stories"] == ["data/stories.yml"]
    assert project_config["config"] == "config.yml"
    assert project_config["test_data"] == ["tests/"]
    assert project_config["train_test_split"] == "train_test_split/"
    assert project_config["results"] == "results/"


def test_no_project_file_and_legacy_config_file_with_importers(
    monkeypatch: MonkeyPatch, tmp_path: Path
):
    """Test missing project file and legacy config with importer works."""


def test_project_file_with_cli_argument(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Test project file with CLI arguments to override works."""


def test_no_project_file_heuristics(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Test heuristics work when project file is missing."""
