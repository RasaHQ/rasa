import os
from pathlib import Path
from typing import Callable
from _pytest.monkeypatch import MonkeyPatch
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


def test_expand_init_dbl_dot_path(run_with_stdin: Callable[..., RunResult]):
    expandable_path = "./workspace/test/../test2"
    expanded_path = os.path.realpath(os.path.expanduser(expandable_path))

    run_with_stdin(
        "init", "--init-dir", expandable_path, stdin=b"\nYN"
    ) 

    assert os.path.isdir(expanded_path)


def test_expand_init_tilde_path(run_with_stdin: Callable[..., RunResult], monkeypatch: MonkeyPatch, tmp_path: Path):

    def mockreturn(path):
        return tmp_path

    monkeypatch.setattr(os.path, 'expanduser', mockreturn)

    expandable_path = "~/workspace"
    expanded_path = os.path.realpath(os.path.expanduser(expandable_path))


    run_with_stdin(
        "init", "--init-dir", expandable_path, stdin=b"\nN\n"
    ) 

    assert os.path.isdir(expanded_path)

def test_expand_init_vars_path(run_with_stdin: Callable[..., RunResult], monkeypatch: MonkeyPatch, tmp_path: Path):

    def mockreturn(path):
        return tmp_path

    monkeypatch.setattr(os.path, 'expandvars', mockreturn)

    expandable_path = "$HOME/workspace"
    expanded_path = os.path.realpath(os.path.expanduser(expandable_path))

    run_with_stdin(
        "init", "--init-dir", expandable_path, stdin=b"\nN\n"
    ) 

    assert os.path.isdir(expanded_path)


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
