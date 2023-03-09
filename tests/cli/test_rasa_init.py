import argparse
import os
from pathlib import Path
from typing import Callable
from _pytest.pytester import RunResult
from _pytest.monkeypatch import MonkeyPatch

from rasa.cli import scaffold
from tests.conftest import enable_cache
from tests.core.channels.test_cmdline import mock_stdin

from tests.cli.conftest import RASA_EXE


def test_init_using_init_dir_option(run_with_stdin: Callable[..., RunResult]):
    os.makedirs("./workspace")
    run_with_stdin(
        "init", "--quiet", "--init-dir", "./workspace", stdin=b"N"
    )  # avoid training an initial model

    required_files = [
        "actions/__init__.py",
        "actions/actions.py",
        "domain.yml",
        "config.yml",
        "credentials.yml",
        "endpoints.yml",
        "data/nlu.yml",
        "data/stories.yml",
        "data/rules.yml",
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
        [--init-dir INIT_DIR]"""

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
            "language": "en",
            "pipeline": [{"name": "KeywordIntentClassifier"}],
            "policies": [{"name": "RulePolicy"}],
            "recipe": "default.v1",
        }

    monkeypatch.setattr(
        "rasa.shared.importers.importer.CombinedDataImporter.get_config",
        mock_get_config,
    )
    # Cache dir is auto patched to be a temp directory, this makes it
    # go back to local project folder so we can test it is created correctly.
    with enable_cache(Path(".rasa", "cache")):
        mock_stdin([])
        scaffold.init_project(args, str(new_project_folder_path))
    assert os.getcwd() == str(new_project_folder_path)
    assert os.path.exists(".rasa/cache")
