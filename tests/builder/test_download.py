import asyncio
import io
import sys
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder import download
from rasa.builder.constants import MAX_BACKUP_SIZE
from rasa.builder.exceptions import ProjectGenerationError


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


def test_create_bot_project_archive_adds_all_files(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setenv("RASA_PRO_LICENSE", "testlicense")
    bot_files = {"config.yml": "config", "domain.yml": "domain"}
    archive = download.create_bot_project_archive(bot_files, "projid", tmp_path)
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


def test_create_bot_project_archive_includes_copilot_db_if_exists(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that copilot database is included in archive when it exists."""
    monkeypatch.setenv("RASA_PRO_LICENSE", "testlicense")

    # Create a mock copilot database file
    copilot_db_dir = tmp_path / ".rasa"
    copilot_db_dir.mkdir(parents=True)
    copilot_db_path = copilot_db_dir / "copilot.db"
    copilot_db_content = b"mock copilot database content"
    copilot_db_path.write_bytes(copilot_db_content)

    bot_files = {"config.yml": "config", "domain.yml": "domain"}
    archive = download.create_bot_project_archive(bot_files, "projid", tmp_path)
    tar_buffer = io.BytesIO(archive)

    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        names = tar.getnames()
        # Check that copilot db is included
        assert ".rasa/copilot.db" in names

        # Verify the content
        db_file = tar.extractfile(".rasa/copilot.db")
        assert db_file.read() == copilot_db_content


def test_create_bot_project_archive_without_copilot_db(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that archive is created successfully when copilot db doesn't exist."""
    monkeypatch.setenv("RASA_PRO_LICENSE", "testlicense")
    bot_files = {"config.yml": "config", "domain.yml": "domain"}

    archive = download.create_bot_project_archive(bot_files, "projid", tmp_path)
    tar_buffer = io.BytesIO(archive)

    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        names = tar.getnames()
        # Should include all other files
        for fname in [
            "config.yml",
            "domain.yml",
            ".env",
            ".python-version",
            "pyproject.toml",
            "README.md",
        ]:
            assert fname in names
        # Copilot db should not be present
        assert ".rasa/copilot.db" not in names


def test_create_bot_project_archive_includes_git_dir_if_exists(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that .git directory is included in archive when it exists."""
    monkeypatch.setenv("RASA_PRO_LICENSE", "testlicense")

    # Create a mock .git directory with some files
    git_dir = tmp_path / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "config").write_text("mock git config")
    (git_dir / "HEAD").write_text("ref: refs/heads/main")

    # Create subdirectories
    refs_dir = git_dir / "refs" / "heads"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abc123def456")

    objects_dir = git_dir / "objects"
    objects_dir.mkdir(parents=True)
    (objects_dir / "pack").mkdir()
    (objects_dir / "info").mkdir()

    bot_files = {"config.yml": "config", "domain.yml": "domain"}
    archive = download.create_bot_project_archive(bot_files, "projid", tmp_path)
    tar_buffer = io.BytesIO(archive)

    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        names = tar.getnames()
        # Check that .git directory and its files are included
        assert ".git" in names
        assert ".git/config" in names
        assert ".git/HEAD" in names
        assert ".git/refs/heads/main" in names

        # Verify the content of git files
        config_file = tar.extractfile(".git/config")
        assert config_file.read().decode() == "mock git config"

        head_file = tar.extractfile(".git/HEAD")
        assert head_file.read().decode() == "ref: refs/heads/main"


def test_create_bot_project_archive_without_git_dir(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Test that archive is created successfully when .git doesn't exist."""
    monkeypatch.setenv("RASA_PRO_LICENSE", "testlicense")
    bot_files = {"config.yml": "config", "domain.yml": "domain"}

    archive = download.create_bot_project_archive(bot_files, "projid", tmp_path)
    tar_buffer = io.BytesIO(archive)

    with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
        names = tar.getnames()
        # Should include all other files
        for fname in [
            "config.yml",
            "domain.yml",
            ".env",
            ".python-version",
            "pyproject.toml",
            "README.md",
        ]:
            assert fname in names
        # .git directory should not be present
        git_files = [name for name in names if name.startswith(".git")]
        assert len(git_files) == 0


def test_valid_s3_url_standard_region() -> None:
    url = "https://my-bucket.s3.us-east-1.amazonaws.com/path/to/file.tar.gz"
    # Should not raise any exception
    download.validate_s3_url(url)


def test_valid_s3_url_global_endpoint() -> None:
    url = "https://my-bucket.s3.amazonaws.com/path/to/file.tar.gz"
    # Should not raise any exception
    download.validate_s3_url(url)


def test_valid_s3_url_different_regions() -> None:
    urls = [
        "https://bucket.s3.eu-west-1.amazonaws.com/file.tar.gz",
        "https://bucket.s3.ap-southeast-1.amazonaws.com/file.tar.gz",
        "https://bucket.s3.ca-central-1.amazonaws.com/file.tar.gz",
        "https://bucket.s3-us-west-2.amazonaws.com/file.tar.gz",
    ]
    for url in urls:
        download.validate_s3_url(url)


def test_valid_s3_url_with_query_params() -> None:
    url = (
        "https://my-bucket.s3.amazonaws.com/file.tar.gz?"
        "X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=..."
    )
    # Should not raise any exception
    download.validate_s3_url(url)


def test_invalid_url_no_hostname() -> None:
    with pytest.raises(ValueError, match="URL must have a valid hostname"):
        download.validate_s3_url("not-a-valid-url")


def test_invalid_url_non_aws_domain() -> None:
    url = "https://evil-website.com/malicious-file.tar.gz"
    with pytest.raises(
        ValueError, match="URL must be from an AWS S3 domain, got: evil-website.com"
    ):
        download.validate_s3_url(url)


def test_invalid_url_missing_s3_in_hostname() -> None:
    url = "https://my-bucket.amazonaws.com/file.tar.gz"
    with pytest.raises(ValueError, match="URL must be from an AWS S3 domain"):
        download.validate_s3_url(url)


def test_invalid_url_not_amazonaws_domain() -> None:
    url = "https://s3.fake-aws.com/file.tar.gz"
    with pytest.raises(
        ValueError, match="URL must be from an AWS S3 domain, got: s3.fake-aws.com"
    ):
        download.validate_s3_url(url)


def test_invalid_url_subdomain_spoofing_attempt() -> None:
    url = "https://s3.amazonaws.com.evil.com/file.tar.gz"
    with pytest.raises(ValueError, match="URL must be from an AWS S3 domain"):
        download.validate_s3_url(url)


def test_valid_s3_url_case_insensitive() -> None:
    url = "https://my-bucket.S3.US-EAST-1.AMAZONAWS.COM/file.tar.gz"
    # Should not raise any exception
    download.validate_s3_url(url)


@pytest.mark.asyncio
async def test_successful_download() -> None:
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"
    test_content = b"test backup content"

    async def mock_chunks(chunk_size):
        yield test_content

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Length": str(len(test_content))}
    mock_response.content.iter_chunked = mock_chunks

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        temp_file_path = await download.download_backup_from_url(url)

        # Verify file was created
        assert Path(temp_file_path).exists()

        # Verify content
        with open(temp_file_path, "rb") as f:
            assert f.read() == test_content

        # Cleanup
        Path(temp_file_path).unlink()


@pytest.mark.asyncio
async def test_download_invalid_url() -> None:
    url = "https://evil-website.com/backup.tar.gz"

    with pytest.raises(ValueError, match="URL must be from an AWS S3 domain"):
        await download.download_backup_from_url(url)


@pytest.mark.asyncio
async def test_download_http_error() -> None:
    """Test that download fails with proper error on HTTP error."""
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"

    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.reason = "Not Found"

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(
            ProjectGenerationError,
            match="Failed to download backup from presigned URL. HTTP 404",
        ):
            await download.download_backup_from_url(url)


@pytest.mark.asyncio
async def test_download_file_too_large_content_length() -> None:
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"
    too_large_size = MAX_BACKUP_SIZE + 1

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Length": str(too_large_size)}

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(
            ProjectGenerationError,
            match="Backup file too large",
        ):
            await download.download_backup_from_url(url)


@pytest.mark.asyncio
async def test_download_file_too_large_during_download() -> None:
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"
    # Create chunks that together exceed the limit
    chunk_size = 10 * 1024 * 1024  # 10MB chunks
    num_chunks = (MAX_BACKUP_SIZE // chunk_size) + 2

    async def mock_chunks(requested_chunk_size):
        for _ in range(num_chunks):
            yield b"x" * chunk_size

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {}  # No Content-Length header
    mock_response.content.iter_chunked = mock_chunks

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(
            ProjectGenerationError,
            match="Backup file too large",
        ):
            temp_file = await download.download_backup_from_url(url)
            # If we somehow get a file path, clean it up
            Path(temp_file).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_download_timeout() -> None:
    """Test that download handles timeout gracefully."""
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock(
        side_effect=asyncio.TimeoutError("Connection timed out")
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(
            ProjectGenerationError,
            match="Download timeout: Presigned URL may have expired",
        ):
            await download.download_backup_from_url(url)


@pytest.mark.asyncio
async def test_download_network_error() -> None:
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    import aiohttp

    mock_session.get = MagicMock(
        side_effect=aiohttp.ClientError("Network connection failed")
    )

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with pytest.raises(
            ProjectGenerationError,
            match="Network error downloading backup",
        ):
            await download.download_backup_from_url(url)


@pytest.mark.asyncio
async def test_download_cleans_up_temp_file_on_error() -> None:
    url = "https://bucket.s3.amazonaws.com/backup.tar.gz"

    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_session.get = MagicMock()
    mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
    mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)

    temp_file_path = None

    # Patch mkstemp to capture the path
    original_mkstemp = download.tempfile.mkstemp

    def mock_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        nonlocal temp_file_path
        temp_file_path = path
        return fd, path

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with patch.object(download.tempfile, "mkstemp", mock_mkstemp):
            with pytest.raises(ProjectGenerationError):
                await download.download_backup_from_url(url)

            # Verify temp file was cleaned up
            if temp_file_path:
                assert not Path(temp_file_path).exists()
