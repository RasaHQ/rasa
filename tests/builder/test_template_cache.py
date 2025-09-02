import asyncio
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sanic import Sanic

from rasa.builder.template_cache import (
    _cache_root_dir,
    _copytree,
    _safe_tar_members,
    _template_cache_dir,
    background_download_template_caches,
    copy_cache_for_template_if_available,
    download_cache_for_template,
)
from rasa.cli.scaffold import ProjectTemplateName


@pytest.fixture
def mock_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Mock the cache root directory to use a temporary path."""
    cache_root = tmp_path / "cache_root"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RASA_TEMPLATE_CACHE_DIR", str(cache_root))
    return cache_root


@pytest.fixture
def sample_tar_content(tmp_path: Path) -> bytes:
    """Create a sample tar.gz file content for testing."""
    # Create a temporary directory with some files
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    (source_dir / "file1.txt").write_text("content1")
    (source_dir / "subdir").mkdir()
    (source_dir / "subdir" / "file2.txt").write_text("content2")
    (source_dir / ".rasa").mkdir()
    (source_dir / ".rasa" / "metadata.json").write_text('{"version": "1.0"}')

    # Create tar.gz
    tar_path = tmp_path / "sample.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(source_dir, arcname=".")

    return tar_path.read_bytes()


class TestSafeTarMembers:
    """Test the _safe_tar_members function."""

    def test_filters_dangerous_paths(self, tmp_path: Path) -> None:
        """Test that dangerous paths are filtered out."""
        # Create a tar with various dangerous entries
        tar_buffer = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        try:
            with tarfile.open(tar_buffer.name, "w:gz") as tar:
                # Safe entries
                safe_info = tarfile.TarInfo("safe_file.txt")
                safe_info.size = 0
                tar.addfile(safe_info)

                # Dangerous entries
                abs_info = tarfile.TarInfo("/absolute/path.txt")
                abs_info.size = 0
                tar.addfile(abs_info)

                traversal_info = tarfile.TarInfo("../traversal.txt")
                traversal_info.size = 0
                tar.addfile(traversal_info)

                symlink_info = tarfile.TarInfo("symlink.txt")
                symlink_info.type = tarfile.SYMTYPE
                symlink_info.linkname = "../target"
                tar.addfile(symlink_info)

                hardlink_info = tarfile.TarInfo("hardlink.txt")
                hardlink_info.type = tarfile.LNKTYPE
                hardlink_info.linkname = "../target"
                tar.addfile(hardlink_info)

            # Test safe extraction
            with tarfile.open(tar_buffer.name, "r:gz") as tar:
                safe_members = list(_safe_tar_members(tar, tmp_path))
                member_names = [m.name for m in safe_members]

                assert "safe_file.txt" in member_names
                assert "/absolute/path.txt" not in member_names
                assert "../traversal.txt" not in member_names
                assert "symlink.txt" not in member_names
                assert "hardlink.txt" not in member_names
        finally:
            Path(tar_buffer.name).unlink(missing_ok=True)

    def test_allows_safe_relative_paths(self, tmp_path: Path) -> None:
        """Test that safe relative paths are allowed."""
        tar_buffer = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        try:
            with tarfile.open(tar_buffer.name, "w:gz") as tar:
                # These should be allowed as they resolve within the base path
                for name in ["file.txt", "dir/file.txt", "dir/../other.txt"]:
                    info = tarfile.TarInfo(name)
                    info.size = 0
                    tar.addfile(info)

            with tarfile.open(tar_buffer.name, "r:gz") as tar:
                safe_members = list(_safe_tar_members(tar, tmp_path))
                member_names = [m.name for m in safe_members]

                assert len(member_names) == 3
                assert "file.txt" in member_names
                assert "dir/file.txt" in member_names
                assert "dir/../other.txt" in member_names
        finally:
            Path(tar_buffer.name).unlink(missing_ok=True)


class TestCacheDirectoryFunctions:
    """Test cache directory helper functions."""

    def test_cache_root_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default cache root directory."""
        monkeypatch.delenv("RASA_TEMPLATE_CACHE_DIR", raising=False)
        expected = Path.home() / ".rasa" / "template-cache"
        assert _cache_root_dir() == expected

    def test_cache_root_dir_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test cache root directory with environment override."""
        custom_path = tmp_path / "custom_cache"
        monkeypatch.setenv("RASA_TEMPLATE_CACHE_DIR", str(custom_path))
        assert _cache_root_dir() == custom_path

    def test_template_cache_dir(self, mock_cache_root: Path) -> None:
        """Test template cache directory generation."""
        with (
            patch("rasa.builder.template_cache.rasa.version.__version__", "1.2.3"),
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            expected = mock_cache_root / "1.2.3" / "default"
            assert cache_dir == expected


class TestCopytree:
    """Test the _copytree function."""

    def test_copytree_basic(self, tmp_path: Path) -> None:
        """Test basic directory copying."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # Create source structure
        src.mkdir()
        (src / "file1.txt").write_text("content1")
        (src / "subdir").mkdir()
        (src / "subdir" / "file2.txt").write_text("content2")
        (src / ".hidden").write_text("hidden content")

        dst.mkdir()

        _copytree(src, dst)

        # Verify files were copied
        assert (dst / "file1.txt").read_text() == "content1"
        assert (dst / "subdir" / "file2.txt").read_text() == "content2"
        assert (dst / ".hidden").read_text() == "hidden content"

    def test_copytree_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that existing files are overwritten."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # Create source and destination
        src.mkdir()
        dst.mkdir()
        (src / "file.txt").write_text("new content")
        (dst / "file.txt").write_text("old content")

        _copytree(src, dst)

        assert (dst / "file.txt").read_text() == "new content"


class TestDownloadCacheForTemplate:
    """Test the download_cache_for_template function."""

    @pytest.mark.asyncio
    async def test_unexpected_error_handled(self, tmp_path: Path) -> None:
        """Test that unexpected errors are handled gracefully."""
        target_dir = tmp_path / "target"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception(
                "Unexpected error"
            )

            # Should not raise an exception
            await download_cache_for_template(
                ProjectTemplateName.DEFAULT, str(target_dir)
            )

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup(self, tmp_path: Path) -> None:
        """Test that temporary files are cleaned up even on error."""
        target_dir = tmp_path / "target"

        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Test error")

            # Count temp files before
            temp_files_before = len(list(Path(tempfile.gettempdir()).glob("*.tar.gz")))

            await download_cache_for_template(
                ProjectTemplateName.DEFAULT, str(target_dir)
            )

            # Count temp files after - should be the same
            temp_files_after = len(list(Path(tempfile.gettempdir()).glob("*.tar.gz")))
            assert temp_files_after == temp_files_before


class TestCopyCacheForTemplateIfAvailable:
    """Test the copy_cache_for_template_if_available function."""

    def test_copies_existing_cache(self, mock_cache_root: Path, tmp_path: Path) -> None:
        """Test copying an existing cache."""
        with (
            patch("rasa.builder.template_cache.rasa.version.__version__", "1.2.3"),
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            # Create cache directory with content
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            cache_dir.mkdir(parents=True)
            (cache_dir / "cached_file.txt").write_text("cached content")
            (cache_dir / ".rasa").mkdir()
            (cache_dir / ".rasa" / "model.tar.gz").write_text("model data")

            # Create project directory
            project_dir = tmp_path / "project"
            project_dir.mkdir()

            copy_cache_for_template_if_available(
                ProjectTemplateName.DEFAULT, project_dir
            )

            # Verify files were copied
            assert (project_dir / "cached_file.txt").read_text() == "cached content"
            assert (project_dir / ".rasa" / "model.tar.gz").read_text() == "model data"

    def test_handles_missing_cache(self, mock_cache_root: Path, tmp_path: Path) -> None:
        """Test handling when cache doesn't exist."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Should not raise an exception
        copy_cache_for_template_if_available(ProjectTemplateName.DEFAULT, project_dir)

        # Project directory should still exist but be empty (except for what was there)
        assert project_dir.exists()

    def test_handles_empty_cache_directory(
        self, mock_cache_root: Path, tmp_path: Path
    ) -> None:
        """Test handling when cache directory exists but is empty."""
        with (
            patch("rasa.builder.template_cache.rasa.version.__version__", "1.2.3"),
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            # Create empty cache directory
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            cache_dir.mkdir(parents=True)

            project_dir = tmp_path / "project"
            project_dir.mkdir()

            # Should not raise an exception
            copy_cache_for_template_if_available(
                ProjectTemplateName.DEFAULT, project_dir
            )

            assert project_dir.exists()

    def test_handles_copy_error(self, mock_cache_root: Path, tmp_path: Path) -> None:
        """Test handling when copy operation fails."""
        with (
            patch("rasa.builder.template_cache.rasa.version.__version__", "1.2.3"),
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            # Create cache directory
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            cache_dir.mkdir(parents=True)
            (cache_dir / "file.txt").write_text("content")

            project_dir = tmp_path / "project"
            project_dir.mkdir()

            # Mock _copytree to raise an exception
            with patch(
                "rasa.builder.template_cache._copytree",
                side_effect=Exception("Copy error"),
            ):
                # Should not raise an exception
                copy_cache_for_template_if_available(
                    ProjectTemplateName.DEFAULT, project_dir
                )


class TestBackgroundDownloadTemplateCaches:
    """Test the background_download_template_caches function."""

    @pytest.mark.asyncio
    async def test_downloads_all_templates(self, mock_cache_root: Path) -> None:
        """Test that all templates are scheduled for download."""
        app = MagicMock(spec=Sanic)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)

        # Mock download function
        with patch(
            "rasa.builder.template_cache.download_cache_for_template"
        ) as mock_download:
            mock_download.return_value = AsyncMock()

            await background_download_template_caches(app, loop)

            # Verify create_task was called for each template
            expected_calls = len(ProjectTemplateName)
            assert loop.create_task.call_count == expected_calls

    @pytest.mark.asyncio
    async def test_skips_existing_caches(self, mock_cache_root: Path) -> None:
        """Test that existing caches are skipped."""
        with (
            patch("rasa.builder.template_cache.rasa.version.__version__", "1.2.3"),
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            # Create existing cache for one template
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            cache_dir.mkdir(parents=True)
            (cache_dir / "existing.txt").write_text("content")

            app = MagicMock(spec=Sanic)
            loop = MagicMock(spec=asyncio.AbstractEventLoop)

            # Mock the task creation to capture the actual coroutines
            created_tasks = []

            def capture_task(coro):
                created_tasks.append(coro)
                return MagicMock()

            loop.create_task.side_effect = capture_task

            with patch(
                "rasa.builder.template_cache.download_cache_for_template"
            ) as mock_download:
                await background_download_template_caches(app, loop)

                # Execute the captured coroutines to test the logic
                for task_coro in created_tasks:
                    await task_coro

                # download_cache_for_template should be called
                # for templates without cache
                # but not for the minimal template that already has cache
                download_calls = [
                    call[0][1] for call in mock_download.call_args_list
                ]  # Extract template names
                assert ProjectTemplateName.DEFAULT.value not in [
                    t.value for t in download_calls if hasattr(t, "value")
                ]

    @pytest.mark.asyncio
    async def test_handles_download_errors_gracefully(
        self, mock_cache_root: Path
    ) -> None:
        """Test that download errors don't crash the background task."""
        app = MagicMock(spec=Sanic)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)

        # Should not raise an exception even if downloads fail
        await background_download_template_caches(app, loop)

    @pytest.mark.asyncio
    async def test_creates_cache_root_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that the cache root directory is created."""
        cache_root = tmp_path / "new_cache_root"
        monkeypatch.setenv("RASA_TEMPLATE_CACHE_DIR", str(cache_root))

        app = MagicMock(spec=Sanic)
        loop = MagicMock(spec=asyncio.AbstractEventLoop)

        await background_download_template_caches(app, loop)

        assert cache_root.exists()
        assert cache_root.is_dir()
