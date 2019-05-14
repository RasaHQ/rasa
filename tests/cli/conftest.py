import pytest

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


@pytest.fixture
def run(testdir):
    def do_run(*args):
        args = ["rasa"] + list(args)
        return testdir.run(*args), testdir.tmpdir

    return do_run
