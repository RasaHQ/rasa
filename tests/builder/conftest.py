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
