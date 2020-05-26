import os
from typing import Callable
from _pytest.pytester import RunResult


def test_init_using_init_dir_option(run_with_stdin: Callable[..., RunResult]):
    os.makedirs("./workspace")
    run_with_stdin(
        "init", "--quiet", "--init-dir", "./workspace", stdin=b"YN"
    )  # avoid training an initial model

    assert os.path.exists("./workspace/actions.py")
    assert os.path.exists("./workspace/domain.yml")
    assert os.path.exists("./workspace/config.yml")
    assert os.path.exists("./workspace/credentials.yml")
    assert os.path.exists("./workspace/endpoints.yml")
    assert os.path.exists("./workspace/data/nlu.md")
    assert os.path.exists("./workspace/data/stories.md")


def test_not_fount_init_path(run: Callable[..., RunResult]):
    output = run("init", "--no-prompt", "--quiet", "--init-dir", "./workspace")

    assert (
        output.outlines[-1]
        == "\033[91mProject init path './workspace' not found.\033[0m"
    )


def test_init_help(run: Callable[..., RunResult]):
    output = run("init", "--help")

    assert (
        output.outlines[0]
        == "usage: rasa init [-h] [-v] [-vv] [--quiet] [--no-prompt] [--init-dir INIT_DIR]"
    )


def test_user_asked_to_train_model(run_with_stdin: Callable[..., RunResult]):
    run_with_stdin("init", stdin=b"\nYN")
    assert not os.path.exists("models")
