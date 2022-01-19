import sys
import importlib
import pytest

import rasa.core.constants as constants
import rasa.utils.endpoints as endpoint_utils


@pytest.fixture
def custom_request_timeout_module():
    # set --request-timeout cmdline arg and
    # reload modules to reflect changes
    sys.argv.extend(["--request-timeout", "40"])
    importlib.reload(constants)
    yield importlib.reload(endpoint_utils)

    # Undo cmdline args change and reset affected modules
    arg_index = sys.argv.index("--request-timeout")
    del sys.argv[arg_index]
    del sys.argv[arg_index]
    importlib.reload(constants)
    importlib.reload(endpoint_utils)


@pytest.fixture
def default_request_timeout_module():
    yield endpoint_utils


async def test_custom_timeout(custom_request_timeout_module):
    """
    Verify that DEFAULT_REQUEST_TIMEOUT can be set
    overriding the default
    """
    conf = custom_request_timeout_module.read_endpoint_config(
        "data/test_endpoints/example_endpoints.yml", "tracker_store"
    )

    async with conf.session() as client_session:
        assert client_session.timeout.total == 40


async def test_default_timeout(default_request_timeout_module):
    """
    Verify the DEFAULT_REQUEST_TIMEOUT is the default
    value when not overridden
    """
    conf = default_request_timeout_module.read_endpoint_config(
        "data/test_endpoints/example_endpoints.yml", "tracker_store"
    )

    async with conf.session() as client_session:
        assert client_session.timeout.total == (60 * 5)
