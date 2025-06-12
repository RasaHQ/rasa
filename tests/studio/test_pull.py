from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import questionary

import rasa.studio.download.download
import rasa.studio.link
import rasa.studio.pull
from rasa.shared.importers.importer import TrainingDataImporter


def test_pull_config_only_writes_file(project: Path, monkeypatch: pytest.MonkeyPatch):
    handler = MagicMock()
    handler.get_config.return_value = "language: en\npipeline: []"
    monkeypatch.setattr(rasa.studio.pull, "StudioDataHandler", lambda *_, **__: handler)

    args = Namespace(config="config.yml")
    rasa.studio.pull.handle_pull_config(args)

    written = (project / "config.yml").read_text()
    assert written.startswith("language: en")


def test_pull_endpoints_only_writes_file(
    project: Path, monkeypatch: pytest.MonkeyPatch
):
    handler = MagicMock()
    handler.get_endpoints.return_value = "nlg:\n  type: utter"
    monkeypatch.setattr(rasa.studio.pull, "StudioDataHandler", lambda *_, **__: handler)

    args = Namespace(endpoints="endpoints.yml")
    rasa.studio.pull.handle_pull_endpoints(args)

    written = (project / "endpoints.yml").read_text()
    assert "type: utter" in written


def test_pull_all_creates_files(project: Path, monkeypatch: pytest.MonkeyPatch):
    handler = MagicMock()
    handler.has_nlu.return_value = False
    handler.get_config.return_value = "some config text"
    handler.get_endpoints.return_value = "some endpoints text"
    monkeypatch.setattr(
        rasa.studio.download.download, "StudioDataHandler", lambda *_, **__: handler
    )

    data_from_studio = MagicMock(spec=TrainingDataImporter)
    data_local = MagicMock(spec=TrainingDataImporter)
    mock_import_data = MagicMock(return_value=(data_from_studio, data_local))
    monkeypatch.setattr(
        "rasa.studio.download.download.import_data_from_studio", mock_import_data
    )
    monkeypatch.setattr(questionary, "confirm", MagicMock(ask=lambda: "y"))

    args = Namespace(
        domain="domain.yml",
        data="data",
        overwrite=False,
        config="config.yml",
        endpoints="endpoints.yml",
    )
    rasa.studio.pull.handle_pull(args)

    assert "some config text" in (project / "config.yml").read_text()
    assert "some endpoints text" in (project / "endpoints.yml").read_text()
