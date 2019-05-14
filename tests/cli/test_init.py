import pytest

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


@pytest.fixture
def run(testdir):
    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir._run(*args)

    return do_run


def test_init(tmpdir, run):
    run("init", "--no-prompt")


def test_init_help(run):
    help = run("init", "--help")

    assert (
        help.outlines[0] == "usage: rasa init [-h] [-v] [-vv] [--quiet] [--no-prompt]"
    )
