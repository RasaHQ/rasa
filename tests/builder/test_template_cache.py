from pathlib import Path
from unittest.mock import patch

import pytest

from rasa.builder.template_cache import (
    _copytree,
    _template_cache_dir,
    copy_cache_for_template_if_available,
)
from rasa.cli.scaffold import ProjectTemplateName


@pytest.fixture
def mock_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Mock the cache root directory to use a temporary path."""
    cache_root = tmp_path / "cache_root"
    cache_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("RASA_TEMPLATE_CACHE_DIR", str(cache_root))
    return cache_root


class TestCacheDirectoryFunctions:
    """Test cache directory helper functions."""

    def test_template_cache_dir(self, mock_cache_root: Path) -> None:
        """Test template cache directory generation."""
        with (
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            expected = mock_cache_root / "default"
            assert cache_dir == expected


class TestCopytree:
    """Test the _copytree function."""

    async def test_copytree_basic(self, tmp_path: Path) -> None:
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

        await _copytree(src, dst)

        # Verify files were copied
        assert (dst / "file1.txt").read_text() == "content1"
        assert (dst / "subdir" / "file2.txt").read_text() == "content2"
        assert (dst / ".hidden").read_text() == "hidden content"

    async def test_copytree_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that existing files are overwritten."""
        src = tmp_path / "src"
        dst = tmp_path / "dst"

        # Create source and destination
        src.mkdir()
        dst.mkdir()
        (src / "file.txt").write_text("new content")
        (dst / "file.txt").write_text("old content")

        await _copytree(src, dst)

        assert (dst / "file.txt").read_text() == "new content"


class TestCopyCacheForTemplateIfAvailable:
    """Test the copy_cache_for_template_if_available function."""

    async def test_copies_existing_cache(
        self, mock_cache_root: Path, tmp_path: Path
    ) -> None:
        """Test copying an existing cache."""
        with (
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

            await copy_cache_for_template_if_available(
                ProjectTemplateName.DEFAULT, project_dir
            )

            # Verify files were copied
            assert (project_dir / "cached_file.txt").read_text() == "cached content"
            assert (project_dir / ".rasa" / "model.tar.gz").read_text() == "model data"

    async def test_handles_missing_cache(
        self, mock_cache_root: Path, tmp_path: Path
    ) -> None:
        """Test handling when cache doesn't exist."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Should not raise an exception
        await copy_cache_for_template_if_available(
            ProjectTemplateName.DEFAULT, project_dir
        )

        # Project directory should still exist but be empty (except for what was there)
        assert project_dir.exists()

    async def test_handles_empty_cache_directory(
        self, mock_cache_root: Path, tmp_path: Path
    ) -> None:
        """Test handling when cache directory exists but is empty."""
        with (
            patch("rasa.builder.template_cache._CACHE_ROOT_DIR", mock_cache_root),
        ):
            # Create empty cache directory
            cache_dir = _template_cache_dir(ProjectTemplateName.DEFAULT)
            cache_dir.mkdir(parents=True)

            project_dir = tmp_path / "project"
            project_dir.mkdir()

            # Should not raise an exception
            await copy_cache_for_template_if_available(
                ProjectTemplateName.DEFAULT, project_dir
            )

            assert project_dir.exists()

    async def test_handles_copy_error(
        self, mock_cache_root: Path, tmp_path: Path
    ) -> None:
        """Test handling when copy operation fails."""
        with (
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
                await copy_cache_for_template_if_available(
                    ProjectTemplateName.DEFAULT, project_dir
                )
