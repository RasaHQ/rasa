from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.utils.yaml import read_yaml
from rasa.studio.constants import DOMAIN_FILENAME
from rasa.studio.download import (
    _handle_config,
    _handle_domain,
    _handle_endpoints,
    _prepare_target_directory,
    handle_download,
)
from rasa.studio.pull.data import STUDIO_FLOWS_DIR_NAME
from rasa.studio.pull.pull import _prepare_data_and_domain_paths


def test_prepare_data_and_domain_paths_no_domain(mock_args: MagicMock):
    """Test domain and data paths are prepared correctly when no domain is provided."""
    domain_path, data_path = _prepare_data_and_domain_paths(mock_args)
    assert domain_path.exists()
    assert domain_path.is_file()
    assert data_path.is_dir()


def test_handle_download(
    tmp_path: Path,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    assistant_name = "my_assistant"
    args = Namespace(assistant_name=assistant_name)
    handle_download(args)

    target_dir = tmp_path / assistant_name
    assert target_dir.exists() and target_dir.is_dir()

    # config.yml
    studio_config = mock_studio_handler.get_config.return_value
    downloaded_config = (target_dir / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")
    assert downloaded_config == studio_config

    # endpoints.yml
    studio_endpoints = mock_studio_handler.get_endpoints.return_value
    downloaded_endpoints = (target_dir / DEFAULT_ENDPOINTS_PATH).read_text(
        encoding="utf-8"
    )
    assert downloaded_endpoints == studio_endpoints

    # domain.yml
    studio_domain = mock_studio_handler.domain
    downloaded_domain = (target_dir / DOMAIN_FILENAME).read_text(encoding="utf-8")
    assert downloaded_domain == studio_domain

    # flows/add_contact.yml and flows/list_contacts.yml
    studio_flows = mock_studio_handler.flows
    flows = read_yaml(studio_flows)

    flow_ids = ["add_contact", "list_contacts"]
    for flow_id in flow_ids:
        add_contact_flow = flows["flows"][flow_id]
        downloaded_flow_path = (
            target_dir / DEFAULT_DATA_PATH / STUDIO_FLOWS_DIR_NAME / f"{flow_id}.yml"
        )
        downloaded_flow = downloaded_flow_path.read_text(encoding="utf-8")
        assert add_contact_flow == read_yaml(downloaded_flow)["flows"][flow_id]


def test_handle_config_success(
    tmp_path: Path,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    _handle_config(mock_studio_handler, tmp_path)

    config_path = tmp_path / DEFAULT_CONFIG_PATH
    assert config_path.is_file()

    studio_config = mock_studio_handler.get_config.return_value
    downloaded_config = config_path.read_text(encoding="utf-8")
    assert downloaded_config == studio_config


def test_handle_config_missing(tmp_path: Path, mock_studio_handler: MagicMock):
    mock_studio_handler.get_config.return_value = ""

    with pytest.raises(SystemExit):
        _handle_config(mock_studio_handler, tmp_path)


def test_handle_endpoints_success(
    tmp_path: Path,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    _handle_endpoints(mock_studio_handler, tmp_path)

    endpoints_path = tmp_path / DEFAULT_ENDPOINTS_PATH
    assert endpoints_path.is_file()

    studio_config = mock_studio_handler.get_endpoints.return_value
    downloaded_config = endpoints_path.read_text(encoding="utf-8")
    assert downloaded_config == studio_config


def test_handle_domain_success(tmp_path: Path, mock_studio_handler: MagicMock):
    _handle_domain(mock_studio_handler, tmp_path)

    domain_path = tmp_path / DOMAIN_FILENAME
    assert domain_path.is_file()

    studio_domain = mock_studio_handler.domain
    downloaded_domain = domain_path.read_text(encoding="utf-8")
    assert downloaded_domain == studio_domain


def test_handle_endpoints_missing(
    tmp_path: Path,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)
    _handle_domain(mock_studio_handler, tmp_path)

    domain_path = tmp_path / DOMAIN_FILENAME
    assert domain_path.is_file()

    studio_domain = mock_studio_handler.domain
    downloaded_domain = domain_path.read_text(encoding="utf-8")
    assert downloaded_domain == studio_domain


def test_handle_download_persists_flows(
    tmp_path: Path,
    mock_studio_handler: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    args = Namespace(assistant_name="my_assistant")
    handle_download(args)
    flows_dir = tmp_path / "my_assistant" / DEFAULT_DATA_PATH / STUDIO_FLOWS_DIR_NAME
    assert flows_dir.exists()

    studio_flows = mock_studio_handler.flows
    flows = read_yaml(studio_flows)

    flow_ids = ["add_contact", "list_contacts"]
    for flow_id in flow_ids:
        add_contact_flow = flows["flows"][flow_id]

        downloaded_flow_path = flows_dir / f"{flow_id}.yml"
        assert downloaded_flow_path.exists()

        downloaded_flow = downloaded_flow_path.read_text(encoding="utf-8")
        assert add_contact_flow == read_yaml(downloaded_flow)["flows"][flow_id]


def test_prepare_target_directory_creates_new(tmp_path: Path, monkeypatch: MagicMock):
    assistant_name = tmp_path / "my_assistant"
    result_path = _prepare_target_directory(str(assistant_name))
    assert result_path == assistant_name
    assert assistant_name.exists() and assistant_name.is_dir()


def test_prepare_target_directory_cancel_overwrite(
    tmp_path: Path, monkeypatch: MagicMock
):
    assistant_name = tmp_path / "my_assistant"
    assistant_name.mkdir()

    # Pretend user answers “no”
    monkeypatch.setattr(
        "rasa.studio.download.questionary.confirm",
        lambda *_: MagicMock(ask=lambda: False),
    )

    with pytest.raises(SystemExit):
        _prepare_target_directory(str(assistant_name))

    assert assistant_name.exists()


def test_prepare_target_directory_overwrite(tmp_path: Path, monkeypatch):
    assistant_name = tmp_path / "my_assistant"
    assistant_name.mkdir()

    file_to_overwrite = assistant_name / "keep_me.txt"
    file_to_overwrite.touch()

    monkeypatch.setattr(
        "rasa.studio.download.questionary.confirm",
        lambda *_: MagicMock(ask=lambda: True),
    )

    result_path = _prepare_target_directory(str(assistant_name))

    assert result_path == assistant_name
    assert assistant_name.exists() and assistant_name.is_dir()
    assert not file_to_overwrite.exists()
