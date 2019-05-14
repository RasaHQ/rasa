import os


def test_init(run):
    result, temp_dir = run("init", "--no-prompt", "--quiet")

    assert os.path.exists(os.path.join(temp_dir, "actions.py"))
    assert os.path.exists(os.path.join(temp_dir, "domain.yml"))
    assert os.path.exists(os.path.join(temp_dir, "config.yml"))
    assert os.path.exists(os.path.join(temp_dir, "credentials.yml"))
    assert os.path.exists(os.path.join(temp_dir, "endpoints.yml"))
    assert os.path.exists(os.path.join(temp_dir, "models"))
    assert os.path.exists(os.path.join(temp_dir, "data", "nlu.md"))
    assert os.path.exists(os.path.join(temp_dir, "data", "stories.md"))


def test_init_help(run):
    help, _ = run("init", "--help")

    assert (
        help.outlines[0] == "usage: rasa init [-h] [-v] [-vv] [--quiet] [--no-prompt]"
    )
