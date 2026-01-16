import asyncio
import os
from unittest.mock import patch

import pytest
from sanic import Sanic

from rasa.builder.service import bp


@pytest.fixture(autouse=True, scope="session")
def disable_telemetry():
    """Disable telemetry for all tests."""
    # Disable Langfuse telemetry
    os.environ["LANGFUSE_ENABLED"] = "false"
    # Disable OpenTelemetry
    os.environ["OTEL_SDK_DISABLED"] = "true"


@pytest.fixture(autouse=True)
def mock_langfuse_decorator():
    """Mock langfuse.observe decorator to prevent any telemetry."""
    with patch("langfuse.observe", lambda *args, **kwargs: lambda f: f):
        yield


@pytest.fixture()
def sanic_app() -> Sanic:
    app = Sanic("bot_builder_test")
    app.blueprint(bp)
    return app


@pytest.fixture()
def default_event_loop_policy():
    """Ensure default event loop policy is used to avoid uvloop issues with subprocess.

    This fixture ensures that asyncio.create_subprocess_exec works correctly
    by temporarily setting the event loop policy to DefaultEventLoopPolicy.
    uvloop doesn't support child watchers required by create_subprocess_exec.
    """
    original_policy = asyncio.get_event_loop_policy()
    try:
        # Set default policy to avoid uvloop NotImplementedError with subprocess
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        yield
    finally:
        # Restore original policy
        asyncio.set_event_loop_policy(original_policy)
