import pathlib

import rasa.core.run
from rasa.core.agent import Agent
from rasa.core.channels.development_inspector import (
    INSPECT_TEMPLATE_PATH,
    DevelopmentInspectProxy,
)
from rasa.core.channels.rest import RestInput

ABSOLUTE_INSPECT_FOLDER_PATH = (
    pathlib.Path(__file__).parent.parent.parent.parent
    / "rasa"
    / "core"
    / "channels"
    / INSPECT_TEMPLATE_PATH
)

ABSOLUTE_INSPECT_TEMPLATE_PATH = ABSOLUTE_INSPECT_FOLDER_PATH / "index.html"


def test_inspect_html_path() -> None:
    channel = DevelopmentInspectProxy(RestInput.from_credentials({}))
    assert channel.inspect_html_path() == str(ABSOLUTE_INSPECT_FOLDER_PATH)


def test_blueprint_inspect() -> None:
    input_channel = DevelopmentInspectProxy(RestInput.from_credentials({}))

    app = rasa.core.run.configure_app([input_channel], port=5004)
    app.ctx.agent = Agent()
    _, res = app.test_client.get("/webhooks/rest/inspect.html")

    assert res.status_code == 200
    # binary comparison to be platform-agnostic
    with open(ABSOLUTE_INSPECT_TEMPLATE_PATH, mode="rb") as handle:
        assert res.body == handle.read()
