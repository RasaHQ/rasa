import argparse
import os
from pathlib import Path
from typing import Callable

import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult

from rasa.cli import scaffold
from rasa.constants import RASA_DIR_NAME
from rasa.shared.constants import (
    CONFIG_LANGUAGE_KEY,
    CONFIG_PIPELINE_KEY,
    CONFIG_POLICIES_KEY,
    CONFIG_RECIPE_KEY,
)
from tests.cli.conftest import RASA_EXE
from tests.conftest import enable_cache
from tests.core.channels.test_cmdline import mock_stdin


def test_init_using_init_dir_option(run_with_stdin: Callable[..., RunResult]):
    os.makedirs("./workspace")
    run_with_stdin(
        "init", "--quiet", "--init-dir", "./workspace", stdin=b"N"
    )  # avoid training an initial model

    required_files = [
        "actions/__init__.py",
        "actions/action_template.py",
        "actions/add_contact.py",
        "actions/list_contacts.py",
        "actions/remove_contact.py",
        "data/flows/add_contact.yml",
        "data/flows/list_contacts.yml",
        "data/flows/remove_contact.yml",
        "db/contacts.json",
        "domain/add_contact.yml",
        "domain/list_contacts.yml",
        "domain/remove_contact.yml",
        "domain/shared.yml",
        "config.yml",
        "credentials.yml",
        "endpoints.yml",
    ]
    assert all((Path("workspace") / file).exists() for file in required_files)

    # ./__init__.py does not exist anymore
    assert not (Path("workspace") / "__init__.py").exists()


def test_not_found_init_path(run: Callable[..., RunResult]):
    output = run("init", "--no-prompt", "--quiet", "--init-dir", "./workspace")

    assert "Project init path './workspace' not found" in output.outlines[-1]


def test_init_help(run: Callable[..., RunResult]):
    output = run("init", "--help")

    help_text = f"""usage: {RASA_EXE} init [-h] [-v] [-vv] [--quiet]
        [--logging-config-file LOGGING_CONFIG_FILE] [--no-prompt]
        [--init-dir INIT_DIR]
        [--template {{default,tutorial,basic,finance,telco}}]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_user_asked_to_train_model(run_with_stdin: Callable[..., RunResult]):
    run_with_stdin("init", stdin=b"\nYN")
    assert not os.path.exists("models")


def test_train_data_in_project_dir(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Test cache directory placement.

    Tests cache directories for training data are in project root, not
    where `rasa init` is run.
    """
    # We would like to test CLI but can't run it with popen because we want
    # to be able to monkeypatch it. Solution is to call functions inside CLI
    # module. Initial project folder should have been created before
    # `init_project`, that's what we do here.
    monkeypatch.chdir(tmp_path)
    new_project_folder_path = tmp_path / "new-project-folder"
    new_project_folder_path.mkdir()

    # Simulate CLI run arguments.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    scaffold.add_subparser(subparsers, parents=[])

    args = parser.parse_args(["init", "--no-prompt"])

    # Simple config which should train fast.
    def mock_get_config(*args):
        return {
            CONFIG_LANGUAGE_KEY: "en",
            CONFIG_PIPELINE_KEY: [{"name": "KeywordIntentClassifier"}],
            CONFIG_POLICIES_KEY: [{"name": "RulePolicy"}],
            CONFIG_RECIPE_KEY: "default.v1",
        }

    monkeypatch.setattr(
        "rasa.shared.importers.importer.CombinedDataImporter.get_config",
        mock_get_config,
    )
    # Cache dir is auto patched to be a temp directory, this makes it
    # go back to local project folder so we can test it is created correctly.
    with enable_cache(Path(RASA_DIR_NAME, "cache")):
        mock_stdin([])
        # file deepcode ignore PT/test: Test function, seems unlikely to be
        # exploitable in any real sense.
        scaffold.init_project(args, str(new_project_folder_path))
    assert os.getcwd() == str(new_project_folder_path)
    assert os.path.exists(".rasa/cache")


@pytest.mark.parametrize("template", ["default", "tutorial", "plain", "finance"])
def test_train_data_non_default_template(
    run_with_stdin: Callable[..., RunResult],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    template: str,
):
    run_with_stdin(
        "init", "--quiet", "--init-dir", str(tmp_path), stdin=b"N"
    )  # avoid training an initial model

    # picking domain as it is present in all templates
    assert (tmp_path / "domain").exists()
