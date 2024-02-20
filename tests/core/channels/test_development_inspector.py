import pathlib
from typing import Any

import pytest
import rasa.core.run
from rasa.core.agent import Agent

from rasa.core.channels.development_inspector import (
    INSPECT_TEMPLATE_PATH,
    DevelopmentInspectInput,
)

ABSOLUTE_INSPECT_FOLDER_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "rasa"
    / "core"
    / "channels"
    / INSPECT_TEMPLATE_PATH
)

ABSOLUTE_INSPECT_TEMPLATE_PATH = ABSOLUTE_INSPECT_FOLDER_PATH / "index.html"


@pytest.mark.parametrize(
    "credentials,expected_session_persistence",
    [
        ({}, True),
        # cannot be overridden
        ({"session_persistence": False}, True),
    ],
)
def test_from_credentials(credentials: Any, expected_session_persistence: bool) -> None:
    input_channel = DevelopmentInspectInput.from_credentials(credentials)
    assert isinstance(input_channel, DevelopmentInspectInput)
    assert input_channel.session_persistence is expected_session_persistence


def test_inspect_html_path() -> None:
    assert DevelopmentInspectInput.inspect_html_path() == str(
        ABSOLUTE_INSPECT_FOLDER_PATH
    )


def test_blueprint_inspect() -> None:
    input_channel = DevelopmentInspectInput.from_credentials({})

    app = rasa.core.run.configure_app([input_channel], port=5004)
    app.ctx.agent = Agent()
    _, res = app.test_client.get("/webhooks/inspector/inspect.html")

    assert res.status_code == 200
    # binary comparison to be platform-agnostic
    with open(ABSOLUTE_INSPECT_TEMPLATE_PATH, mode="rb") as handle:
        assert res.body == handle.read()
