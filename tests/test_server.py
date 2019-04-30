# -*- coding: utf-8 -*-
import pytest

import rasa
import rasa.constants


@pytest.fixture
def rasa_app(rasa_server):
    return rasa_server.test_client


@pytest.fixture
def rasa_secured_app(rasa_server_secured):
    return rasa_server_secured.test_client


def test_root(rasa_app):
    _, response = rasa_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_root_secured(rasa_secured_app):
    _, response = rasa_secured_app.get("/")
    assert response.status == 200
    assert response.text.startswith("Hello from Rasa:")


def test_version(rasa_app):
    _, response = rasa_app.get("/version")
    content = response.json
    assert response.status == 200
    assert content.get("version") == rasa.__version__
    assert (
        content.get("minimum_compatible_version")
        == rasa.constants.MINIMUM_COMPATIBLE_VERSION
    )


def test_status(rasa_app):
    _, response = rasa_app.get("/status")
    assert response.status == 200
