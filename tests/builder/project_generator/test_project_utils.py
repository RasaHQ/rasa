"""Tests for project generator utility functions."""

from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import pytest

from rasa.builder.project_generator.project_utils import (
    bot_file_paths,
    get_bot_files,
    is_restricted_path,
    path_relative_to_project,
    unsafe_write_to_bot_files,
)
from rasa.utils.io import InvalidPathException


class TestIsRestrictedPath:
    """Test is_restricted_path function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder."""
        return tmp_path

    def test_hidden_file_is_restricted(self, project_folder: Path) -> None:
        """Test that hidden files are restricted."""
        hidden_file = project_folder / ".hidden_file"
        hidden_file.touch()

        assert is_restricted_path(project_folder, hidden_file) is True

    def test_hidden_directory_is_restricted(self, project_folder: Path) -> None:
        """Test that files in hidden directories are restricted."""
        hidden_dir = project_folder / ".git"
        hidden_dir.mkdir()
        file_in_hidden = hidden_dir / "config"
        file_in_hidden.touch()

        assert is_restricted_path(project_folder, file_in_hidden) is True

    def test_pycache_is_restricted(self, project_folder: Path) -> None:
        """Test that __pycache__ directories are restricted."""
        pycache_dir = project_folder / "__pycache__"
        pycache_dir.mkdir()
        pyc_file = pycache_dir / "module.cpython-39.pyc"
        pyc_file.touch()

        assert is_restricted_path(project_folder, pyc_file) is True

    def test_models_directory_restricted_by_default(self, project_folder: Path) -> None:
        """Test that models directory is restricted by default."""
        models_dir = project_folder / "models"
        models_dir.mkdir()
        model_file = models_dir / "model.tar.gz"
        model_file.touch()

        assert is_restricted_path(project_folder, model_file) is True

    def test_models_directory_not_restricted_when_disabled(
        self, project_folder: Path
    ) -> None:
        """Test that models directory is not restricted when flag is disabled."""
        models_dir = project_folder / "models"
        models_dir.mkdir()
        model_file = models_dir / "model.tar.gz"
        model_file.touch()

        assert (
            is_restricted_path(
                project_folder, model_file, exclude_models_directory=False
            )
            is False
        )

    def test_normal_file_not_restricted(self, project_folder: Path) -> None:
        """Test that normal files are not restricted."""
        normal_file = project_folder / "domain.yml"
        normal_file.touch()

        assert is_restricted_path(project_folder, normal_file) is False

    def test_nested_normal_file_not_restricted(self, project_folder: Path) -> None:
        """Test that nested normal files are not restricted."""
        data_dir = project_folder / "data"
        data_dir.mkdir()
        nlu_file = data_dir / "nlu.yml"
        nlu_file.touch()

        assert is_restricted_path(project_folder, nlu_file) is False

    def test_rasa_directory_is_restricted(self, project_folder: Path) -> None:
        """Test that .rasa directory is restricted."""
        rasa_dir = project_folder / ".rasa"
        rasa_dir.mkdir()
        cache_file = rasa_dir / "cache"
        cache_file.touch()

        assert is_restricted_path(project_folder, cache_file) is True


class TestGetBotFiles:
    """Test get_bot_files function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with test files."""
        # Create normal files
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        (tmp_path / "config.yml").write_text("pipeline: []")

        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("nlu:\n  - intent: greet")
        (data_dir / "stories.yml").write_text("stories: []")

        # Create docs directory
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "README.md").write_text("# Documentation")

        # Create restricted directories/files
        (tmp_path / ".hidden").write_text("hidden content")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_text("bytecode")

        models = tmp_path / "models"
        models.mkdir()
        (models / "model.tar.gz").write_text("model data")

        return tmp_path

    def test_get_bot_files_returns_all_non_restricted_files(
        self, project_folder: Path
    ) -> None:
        """Test that get_bot_files returns all non-restricted files."""
        files = get_bot_files(project_folder)

        assert "domain.yml" in files
        assert "config.yml" in files
        assert "data/nlu.yml" in files
        assert "data/stories.yml" in files
        assert "docs/README.md" in files

        # Restricted files should not be included
        assert ".hidden" not in files
        assert "__pycache__/module.pyc" not in files
        assert "models/model.tar.gz" not in files

    def test_get_bot_files_with_extension_filter(self, project_folder: Path) -> None:
        """Test filtering by file extension."""
        files = get_bot_files(project_folder, allowed_file_extensions=["yml", "yaml"])

        assert "domain.yml" in files
        assert "config.yml" in files
        assert "data/nlu.yml" in files
        # md file should be filtered out
        assert "docs/README.md" not in files

    def test_get_bot_files_exclude_docs(self, project_folder: Path) -> None:
        """Test excluding docs directory."""
        files = get_bot_files(project_folder, exclude_docs_directory=True)

        assert "domain.yml" in files
        assert "config.yml" in files
        # Docs should be excluded
        assert "docs/README.md" not in files

    def test_get_bot_files_include_models(self, project_folder: Path) -> None:
        """Test including models directory."""
        files = get_bot_files(project_folder, exclude_models_directory=False)

        assert "domain.yml" in files
        # Models should be included when flag is False
        assert "models/model.tar.gz" in files

    def test_get_bot_files_with_empty_extension_filter(
        self, project_folder: Path
    ) -> None:
        """Test filtering with empty string extension (files without extension)."""
        # Create a file without extension
        (project_folder / "Makefile").write_text("all: build")

        files = get_bot_files(project_folder, allowed_file_extensions=[""])

        assert "Makefile" in files
        assert "domain.yml" not in files

    def test_get_bot_files_handles_read_errors(self, project_folder: Path) -> None:
        """Test that read errors are handled gracefully."""
        # Create a file
        (project_folder / "unreadable.yml").write_text("content")

        # Mock the read_text to raise an exception
        with patch.object(Path, "read_text", side_effect=PermissionError("No access")):
            files = get_bot_files(project_folder)

        # Files with read errors should have None as content
        # (the error is logged and value is set to None)
        assert any(v is None for v in files.values())


class TestBotFilePaths:
    """Test bot_file_paths generator function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder with test files."""
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        (tmp_path / "config.yml").write_text("pipeline: []")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("nlu: []")

        # Restricted
        (tmp_path / ".hidden").write_text("hidden")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "module.pyc").write_text("bytecode")

        return tmp_path

    def test_bot_file_paths_yields_only_non_restricted(
        self, project_folder: Path
    ) -> None:
        """Test that bot_file_paths yields only non-restricted paths."""
        paths = list(bot_file_paths(project_folder))

        file_names = [p.name for p in paths]
        assert "domain.yml" in file_names
        assert "config.yml" in file_names
        assert "nlu.yml" in file_names

        # Restricted should not be yielded
        assert ".hidden" not in file_names
        assert "module.pyc" not in file_names

    def test_bot_file_paths_skips_directories(self, project_folder: Path) -> None:
        """Test that bot_file_paths skips directories."""
        paths = list(bot_file_paths(project_folder))

        # All yielded paths should be files
        assert all(p.is_file() for p in paths)


class TestUnsafeWriteToBotFiles:
    """Test unsafe_write_to_bot_files function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create an empty project folder."""
        return tmp_path

    def test_write_single_file(self, project_folder: Path) -> None:
        """Test writing a single file."""
        files: Dict[str, Optional[str]] = {"domain.yml": "version: '3.1'"}

        unsafe_write_to_bot_files(project_folder, files)

        assert (project_folder / "domain.yml").exists()
        assert (project_folder / "domain.yml").read_text() == "version: '3.1'"

    def test_write_multiple_files(self, project_folder: Path) -> None:
        """Test writing multiple files."""
        files: Dict[str, Optional[str]] = {
            "domain.yml": "version: '3.1'",
            "config.yml": "pipeline: []",
        }

        unsafe_write_to_bot_files(project_folder, files)

        assert (project_folder / "domain.yml").read_text() == "version: '3.1'"
        assert (project_folder / "config.yml").read_text() == "pipeline: []"

    def test_write_creates_parent_directories(self, project_folder: Path) -> None:
        """Test that parent directories are created."""
        files: Dict[str, Optional[str]] = {
            "data/nlu/training.yml": "nlu content",
        }

        unsafe_write_to_bot_files(project_folder, files)

        assert (project_folder / "data" / "nlu" / "training.yml").exists()
        assert (
            project_folder / "data" / "nlu" / "training.yml"
        ).read_text() == "nlu content"

    def test_write_handles_none_content(self, project_folder: Path) -> None:
        """Test that None content creates empty file."""
        files: Dict[str, Optional[str]] = {"empty.yml": None}

        unsafe_write_to_bot_files(project_folder, files)

        assert (project_folder / "empty.yml").exists()
        assert (project_folder / "empty.yml").read_text() == ""

    def test_write_to_hidden_path_raises_error(self, project_folder: Path) -> None:
        """Test that writing to hidden paths raises error by default."""
        files: Dict[str, Optional[str]] = {".hidden/secret.txt": "secret"}

        with pytest.raises(InvalidPathException):
            unsafe_write_to_bot_files(project_folder, files)

    def test_write_to_hidden_path_silent_when_disabled(
        self, project_folder: Path
    ) -> None:
        """Test that hidden paths are silently ignored when flag is disabled."""
        files: Dict[str, Optional[str]] = {
            ".hidden/secret.txt": "secret",
            "normal.yml": "content",
        }

        # Should not raise, just skip the hidden file
        unsafe_write_to_bot_files(project_folder, files, fail_on_restricted_path=False)

        # Hidden file should not be created
        assert not (project_folder / ".hidden" / "secret.txt").exists()
        # Normal file should be created
        assert (project_folder / "normal.yml").exists()

    def test_overwrite_existing_file(self, project_folder: Path) -> None:
        """Test overwriting existing file."""
        # Create existing file
        (project_folder / "domain.yml").write_text("old content")

        files: Dict[str, Optional[str]] = {"domain.yml": "new content"}

        unsafe_write_to_bot_files(project_folder, files)

        assert (project_folder / "domain.yml").read_text() == "new content"


class TestPathRelativeToProject:
    """Test path_relative_to_project function."""

    @pytest.fixture
    def project_folder(self, tmp_path: Path) -> Path:
        """Create a project folder."""
        return tmp_path

    def test_simple_filename(self, project_folder: Path) -> None:
        """Test simple filename returns correct path."""
        result = path_relative_to_project(project_folder, "domain.yml")

        assert result == project_folder / "domain.yml"

    def test_nested_path(self, project_folder: Path) -> None:
        """Test nested path returns correct path."""
        result = path_relative_to_project(project_folder, "data/nlu.yml")

        assert result == project_folder / "data" / "nlu.yml"

    def test_deeply_nested_path(self, project_folder: Path) -> None:
        """Test deeply nested path returns correct path."""
        result = path_relative_to_project(project_folder, "data/nlu/training/en.yml")

        expected = project_folder / "data" / "nlu" / "training" / "en.yml"
        assert result == expected

    def test_path_traversal_prevented(self, project_folder: Path) -> None:
        """Test that path traversal is prevented."""
        # The subpath function should prevent escaping the project folder
        with pytest.raises(InvalidPathException):
            path_relative_to_project(project_folder, "../outside.yml")
