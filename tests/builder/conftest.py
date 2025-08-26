import pytest
from sanic import Sanic

from rasa.builder.service import bp


@pytest.fixture()
def sanic_app() -> Sanic:
    app = Sanic("bot_builder_test")
    app.blueprint(bp)
    return app
