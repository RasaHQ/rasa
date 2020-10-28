import os
from pathlib import Path
from typing import Callable
from _pytest.pytester import RunResult


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

    help_text = (
        "usage: rasa init [-h] [-v] [-vv] [--quiet] [--no-prompt] [--init-dir INIT_DIR]"
    )

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_user_asked_to_train_model(run_with_stdin: Callable[..., RunResult]):
    run_with_stdin("init", stdin=b"\nYN")
    assert not os.path.exists("models")
