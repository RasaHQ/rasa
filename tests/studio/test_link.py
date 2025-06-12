from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import questionary

import rasa.studio.link
from rasa.constants import RASA_DIR_NAME
from rasa.studio.config import StudioConfig


def _studio_config():
    """Return a minimal studio-config stub for the helper."""
    return StudioConfig(
        authentication_server_url="http://auth",
        studio_url="http://studio/graphql",
        realm_name="realm",
        client_id="client",
    )


@pytest.fixture()
def linked_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test inside its own temporary CWD."""
    monkeypatch.chdir(tmp_path)

    # Make sure the project is already linked to an assistant
    _real_read_assistant_name = rasa.studio.link.read_assistant_name

    monkeypatch.setattr(
        rasa.studio.link,
        "read_assistant_name",
        lambda *_, **__: _real_read_assistant_name(tmp_path),
    )
    yield tmp_path


def _mock_studio_ready(monkeypatch: pytest.MonkeyPatch):
    mock_studio_config_cls = MagicMock()
    mock_studio_config_cls.read_config.return_value = _studio_config()
    monkeypatch.setattr(rasa.studio.link, "StudioConfig", mock_studio_config_cls)

    monkeypatch.setattr(rasa.studio.link, "is_auth_working", lambda *_: True)
    monkeypatch.setattr(
        rasa.studio.link, "check_if_assistant_already_exists", lambda *_: True
    )


def test_link_creates_file(linked_project: Path, monkeypatch: pytest.MonkeyPatch):
    _mock_studio_ready(monkeypatch)

    args = Namespace(assistant_name=["my_bot"])
    rasa.studio.link.handle_link(args)

    link_file = linked_project / RASA_DIR_NAME / "studio.yml"
    assert link_file.is_file()

    content = rasa.studio.link._read_link_file(linked_project)
    assert content["assistant_name"] == "my_bot"


def test_link_refuses_overwrite(linked_project: Path, monkeypatch: pytest.MonkeyPatch):
    _mock_studio_ready(monkeypatch)
    link_file = linked_project / RASA_DIR_NAME / "studio.yml"
    link_file.parent.mkdir()
    link_file.write_text("assistant_name: old")

    monkeypatch.setattr(questionary, "confirm", lambda *_: MagicMock(ask=lambda: False))

    args = Namespace(assistant_name=["new_bot"])
    with pytest.raises(SystemExit):
        rasa.studio.link.handle_link(args)


def test_link_file_path(linked_project: Path):
    expected = linked_project / RASA_DIR_NAME / rasa.studio.link._LINK_FILE_NAME
    assert rasa.studio.link._link_file(linked_project) == expected


def test_write_and_read_link_file(linked_project: Path):
    assistant_name = "my_bot"
    studio_url = "http://studio"
    rasa.studio.link._write_link_file(linked_project, assistant_name, studio_url)

    link_file = linked_project / RASA_DIR_NAME / rasa.studio.link._LINK_FILE_NAME
    assert link_file.is_file()

    data = rasa.studio.link._read_link_file(linked_project)
    assert data["assistant_name"] == assistant_name
    assert data["studio_url"] == studio_url


def test_read_assistant_name_success(
    monkeypatch: pytest.MonkeyPatch, linked_project: Path
):
    rasa.studio.link._write_link_file(linked_project, "linked_bot", "dummy")
    assert rasa.studio.link.read_assistant_name() == "linked_bot"


def test_read_assistant_name_not_linked(
    linked_project: Path, monkeypatch: pytest.MonkeyPatch
):
    with pytest.raises(SystemExit):
        rasa.studio.link.read_assistant_name()


def test_get_studio_config_valid(monkeypatch: pytest.MonkeyPatch):
    _mock_studio_ready(monkeypatch)
    result = rasa.studio.link.get_studio_config()
    assert isinstance(result, StudioConfig)
    assert result.studio_url == "http://studio/graphql"


def test_handle_link_creates_assistant_when_missing(
    project: Path, monkeypatch: pytest.MonkeyPatch
):
    mock_upload = MagicMock()
    monkeypatch.setattr(rasa.studio.link, "handle_upload", mock_upload)
    monkeypatch.setattr(questionary, "confirm", MagicMock(ask=lambda: "y"))
    monkeypatch.setattr(
        rasa.studio.link, "check_if_assistant_already_exists", lambda *_: False
    )

    args = Namespace(assistant_name=["brand_new_bot"])
    rasa.studio.link.handle_link(args)

    mock_upload.assert_called_once()
    upload_ns = mock_upload.call_args[0][0]
    assert upload_ns.assistant_name == "brand_new_bot"

    link_file = project / RASA_DIR_NAME / "studio.yml"
    assert link_file.is_file()
    assert "brand_new_bot" in link_file.read_text()


def test_ensure_assistant_exists_nothing_to_do(monkeypatch):
    """Assistant already exists → no prompt, no upload."""
    monkeypatch.setattr(
        rasa.studio.link, "check_if_assistant_already_exists", lambda *_: True
    )
    upload_mock = MagicMock()
    monkeypatch.setattr(rasa.studio.link, "handle_upload", upload_mock)
    confirm_mock = MagicMock()
    monkeypatch.setattr(questionary, "confirm", confirm_mock)

    args = Namespace(assistant_name=["my_bot"])
    rasa.studio.link._ensure_assistant_exists("my_bot", _studio_config(), args)

    upload_mock.assert_not_called()
    confirm_mock.assert_not_called()


def test_ensure_assistant_exists_creates_when_confirmed(monkeypatch):
    """Assistant missing and user confirms → upload happens."""
    monkeypatch.setattr(
        rasa.studio.link, "check_if_assistant_already_exists", lambda *_: False
    )
    monkeypatch.setattr(questionary, "confirm", lambda *_: MagicMock(ask=lambda: True))
    upload_mock = MagicMock()
    monkeypatch.setattr(rasa.studio.link, "handle_upload", upload_mock)

    args = Namespace(assistant_name=["placeholder"])
    rasa.studio.link._ensure_assistant_exists("brand_new_bot", _studio_config(), args)

    upload_mock.assert_called_once()
    assert args.assistant_name == "brand_new_bot"


def test_ensure_assistant_exists_skips_when_declined(monkeypatch):
    """Assistant missing but user declines creation → nothing happens."""
    monkeypatch.setattr(
        rasa.studio.link, "check_if_assistant_already_exists", lambda *_: False
    )
    monkeypatch.setattr(questionary, "confirm", lambda *_: MagicMock(ask=lambda: False))
    upload_mock = MagicMock()
    monkeypatch.setattr(rasa.studio.link, "handle_upload", upload_mock)

    args = Namespace(assistant_name=["whatever"])
    rasa.studio.link._ensure_assistant_exists("new_bot", _studio_config(), args)

    upload_mock.assert_not_called()


def test_handle_link_aborts_when_assistant_missing_and_user_declines_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _mock_studio_ready(monkeypatch)

    # Mock that the assistant does not exist
    monkeypatch.setattr(
        rasa.studio.link, "check_if_assistant_already_exists", lambda *_: False
    )

    # Mock that the user declines the creation of the assistant
    monkeypatch.setattr(questionary, "confirm", lambda *_: MagicMock(ask=lambda: False))

    upload_mock = MagicMock()
    monkeypatch.setattr(rasa.studio.link, "handle_upload", upload_mock)

    args = Namespace(assistant_name=["non_existent_bot"])
    with pytest.raises(SystemExit):
        rasa.studio.link.handle_link(args)
