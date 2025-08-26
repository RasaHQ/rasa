import io
import sys
import tarfile
from typing import Any

from rasa.builder import download


def test_get_env_content(monkeypatch: Any) -> None:
    monkeypatch.setenv("RASA_PRO_LICENSE", "abc123")
    assert download._get_env_content() == "RASA_PRO_LICENSE=abc123\n"


def test_get_python_version_content() -> None:
    expected = f"{sys.version_info.major}.{sys.version_info.minor}\n"
    assert download._get_python_version_content() == expected


def test_get_pyproject_toml_content() -> None:
    project_id = "mybot"
    content = download._get_pyproject_toml_content(project_id)
    assert f'name = "{project_id}"' in content
    assert "[project]" in content


def test_get_readme_content() -> None:
    content = download._get_readme_content()
    assert "# Rasa Assistant" in content
    assert "uv run rasa train" in content


def test_add_file_to_tar_creates_file() -> None:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        download._add_file_to_tar(tar, "foo.txt", "bar")
    tar_buffer.seek(0)
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        names = tar.getnames()
        assert "foo.txt" in names
        member = tar.extractfile("foo.txt")
        assert member.read().decode() == "bar"


def test_create_bot_project_archive_adds_all_files(monkeypatch: Any) -> None:
    monkeypatch.setenv("RASA_PRO_LICENSE", "testlicense")
    bot_files = {"config.yml": "config", "domain.yml": "domain"}
    archive = download.create_bot_project_archive(bot_files, "projid")
    tar_buffer = io.BytesIO(archive)
    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        names = tar.getnames()
        # Should include bot files and all generated files
        for fname in [
            "config.yml",
            "domain.yml",
            ".env",
            ".python-version",
            "pyproject.toml",
            "README.md",
        ]:
            assert fname in names
        # Check .env content
        env_file = tar.extractfile(".env")
        assert env_file.read().decode().startswith("RASA_PRO_LICENSE=testlicense")
