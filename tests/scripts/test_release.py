from pathlib import Path
from typing import Text
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch
from pep440_version_utils import Version

from scripts.release import ask_version, generate_changelog


@pytest.mark.parametrize(
    "user_input,current_version,expected",
    [
        ("major", "2.0.0", "3.0.0"),
        ("minor", "2.1.0", "2.2.0"),
        ("micro", "2.1.1", "2.1.2"),
        ("1.2.3", "2.0.0", "1.2.3"),
        ("rc", "2.0.0", "2.0.1rc1"),
        ("alpha", "2.0.0", "2.0.1a1"),
        ("beta", "2.0.0", "2.0.1b1"),
        ("2.0.0dev1", "1.9.10", "2.0.0dev1"),
    ],
)
def test_ask_version_basic_inputs(
    monkeypatch: MonkeyPatch, user_input: Text, current_version: Text, expected: Text
) -> None:
    mock_get_current = MagicMock(return_value=current_version)
    monkeypatch.setattr("scripts.release.get_current_version", mock_get_current)

    # Mock both text and select questionary methods
    mock_text = MagicMock()
    mock_text.return_value = mock_text
    mock_text.ask.return_value = user_input
    monkeypatch.setattr("scripts.release.questionary.text", mock_text)

    mock_select = MagicMock()
    mock_select.return_value = mock_select
    mock_select.ask.return_value = expected
    monkeypatch.setattr("scripts.release.questionary.select", mock_select)

    result = ask_version()
    assert result == expected


def test_ask_version_prerelease_flow(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.release.get_current_version", MagicMock(return_value="2.0.0")
    )

    mock_text = MagicMock()
    mock_text.return_value = mock_text
    mock_text.ask.return_value = "alpha"
    monkeypatch.setattr("scripts.release.questionary.text", mock_text)

    mock_select = MagicMock()
    mock_select.return_value = mock_select
    mock_select.ask.return_value = "2.1.0a1"
    monkeypatch.setattr("scripts.release.questionary.select", mock_select)

    result = ask_version()
    assert result == "2.1.0a1"
    mock_text.ask.assert_called_once()
    mock_select.ask.assert_called_once()


@pytest.mark.parametrize(
    "version,changelog_exists,expected_calls",
    [
        (Version("1.0.0"), True, 1),
        (Version("2.0.0-alpha.1"), True, 1),
        (Version("3.0.0"), False, 0),
    ],
)
def test_generate_changelog_with_different_versions(
    monkeypatch: MonkeyPatch,
    version: Version,
    changelog_exists: bool,
    expected_calls: int,
    tmp_path: Path,
) -> None:
    mock_check_call = MagicMock()
    monkeypatch.setattr("scripts.release.check_call", mock_check_call)

    if changelog_exists:
        changelog_path = tmp_path / "changelog"
        changelog_path.mkdir()
        monkeypatch.setattr(
            "scripts.release.project_root", MagicMock(return_value=tmp_path)
        )
    else:
        monkeypatch.setattr(
            "scripts.release.project_root", MagicMock(return_value=Path("/nonexistent"))
        )

    generate_changelog(version)

    assert mock_check_call.call_count == expected_calls
    if expected_calls > 0:
        mock_check_call.assert_called_with(
            ["towncrier", "build", "--yes", "--version", str(version)],
            cwd=str(tmp_path),
        )


def test_ask_version_abort(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.release.get_current_version", MagicMock(return_value="1.0.0")
    )

    mock_text = MagicMock()
    mock_text.return_value = mock_text
    mock_text.ask.return_value = None
    monkeypatch.setattr("scripts.release.questionary.text", mock_text)

    with pytest.raises(SystemExit):
        ask_version()


def test_generate_changelog(
    monkeypatch: MonkeyPatch,
) -> None:
    call_mock = MagicMock()
    monkeypatch.setattr("scripts.release.check_call", call_mock)

    generate_changelog(Version("1.0.0"))

    assert call_mock.call_count == 1
    for arg in call_mock.call_args[0][0]:
        assert arg in ["towncrier", "build", "--yes", "--version", "1.0.0"]


def test_generate_changelog_when_path_does_not_exist(
    monkeypatch: MonkeyPatch,
) -> None:
    check_call_mock = MagicMock()
    monkeypatch.setattr("scripts.release.check_call", check_call_mock)
    monkeypatch.setattr(
        "scripts.release.project_root",
        MagicMock(return_value=Path("/path/does/not/exist")),
    )

    generate_changelog(Version("1.0.0"))

    assert check_call_mock.call_count == 0
