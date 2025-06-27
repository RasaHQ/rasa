from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import pytest

import rasa.studio.push
from rasa.studio.results_logger import StudioResult


def _capture_make_request(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    def fake_make_request(endpoint, graphql_req, verify):
        captured["endpoint"] = endpoint
        captured["payload"] = graphql_req
        captured["verify"] = verify
        return StudioResult.success("ok")

    monkeypatch.setattr(rasa.studio.push, "make_request", fake_make_request)
    return captured


def test_handle_push_config(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured = _capture_make_request(monkeypatch)

    args = Namespace(config="config.yml")
    rasa.studio.push.handle_push_config(args)

    inp = captured["payload"]["variables"]["input"]
    assert inp["config"] != ""

    fields = ["endpoints", "flows", "domain", "nlu"]
    for field in fields:
        assert field not in inp


def test_handle_push_endpoints(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured = _capture_make_request(monkeypatch)

    args = Namespace(endpoints="endpoints.yml")
    rasa.studio.push.handle_push_endpoints(args)

    inp = captured["payload"]["variables"]["input"]
    assert inp["endpoints"] != ""

    fields = ["config", "flows", "domain", "nlu"]
    for field in fields:
        assert field not in inp


def test_handle_push_all(
    project: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured = _capture_make_request(monkeypatch)

    monkeypatch.setattr(
        rasa.studio.push.RasaYAMLWriter, "dumps", staticmethod(lambda *_: "nlu_yaml")
    )

    args = Namespace(
        domain="domain.yml",
        data="data",
        config="config.yml",
        endpoints="endpoints.yml",
    )
    rasa.studio.push.handle_push(args)

    inp = captured["payload"]["variables"]["input"]
    assert inp["endpoints"] != ""
    assert inp["config"] != ""
    assert inp["flows"] != ""
    assert inp["domain"] != ""
    assert inp["nlu"] != ""
