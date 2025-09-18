from pathlib import Path

import pytest

from rasa.builder.project_generator import ProjectGenerator
from rasa.utils.io import InvalidPathException


class TestProjectGenerator:
    """Test ProjectGenerator class methods."""

    def test_is_empty_with_empty_directory(self, tmp_path: Path) -> None:
        """Test is_empty returns True for empty directory."""
        generator = ProjectGenerator(tmp_path)
        assert generator.is_empty() is True

    def test_is_empty_with_files(self, tmp_path: Path) -> None:
        """Test is_empty returns False when directory contains files."""
        (tmp_path / "config.yml").write_text("version: '3.1'")
        generator = ProjectGenerator(tmp_path)
        assert generator.is_empty() is False

    def test_is_empty_ignores_hidden_files(self, tmp_path: Path) -> None:
        """Test is_empty ignores hidden files and directories."""
        (tmp_path / ".hidden_file").write_text("hidden")
        (tmp_path / ".hidden_dir").mkdir()
        generator = ProjectGenerator(tmp_path)
        assert generator.is_empty() is True

    def test_is_empty_with_empty_subdirectories(self, tmp_path: Path) -> None:
        """Test is_empty returns True when directory contains only empty subdirs."""
        (tmp_path / "data").mkdir()
        generator = ProjectGenerator(tmp_path)
        # The implementation only checks for files, not directories
        assert generator.is_empty() is True

    def test_is_empty_with_files_in_subdirectories(self, tmp_path: Path) -> None:
        """Test is_empty returns True when files are only in subdirs."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file.txt").write_text("content")
        generator = ProjectGenerator(tmp_path)
        # Should return True because is_empty only checks for files at the root level
        assert generator.is_empty() is True

    def test_is_restricted_path_hidden_files(self, tmp_path: Path) -> None:
        """Test is_restricted_path correctly identifies hidden files."""
        generator = ProjectGenerator(tmp_path)

        # Hidden files should be restricted
        hidden_file = tmp_path / ".hidden_file"
        assert generator.is_restricted_path(hidden_file) is True

        # Hidden directories should be restricted
        hidden_dir = tmp_path / ".hidden_dir"
        assert generator.is_restricted_path(hidden_dir) is True

        # .rasa directory should be restricted
        rasa_dir = tmp_path / ".rasa"
        assert generator.is_restricted_path(rasa_dir) is True

    def test_is_restricted_path_models_directory(self, tmp_path: Path) -> None:
        """Test is_restricted_path correctly identifies models directory."""
        generator = ProjectGenerator(tmp_path)

        # Models directory should be restricted
        models_dir = tmp_path / "models"
        assert generator.is_restricted_path(models_dir) is True

        # Files in models directory should be restricted
        models_file = tmp_path / "models" / "model.tar.gz"
        assert generator.is_restricted_path(models_file) is True

    def test_is_restricted_path_pycache_directory(self, tmp_path: Path) -> None:
        """Test is_restricted_path correctly identifies __pycache__ directory."""
        generator = ProjectGenerator(tmp_path)

        # __pycache__ directory should be restricted
        pycache_dir = tmp_path / "actions" / "__pycache__"
        assert generator.is_restricted_path(pycache_dir) is True

    def test_is_restricted_path_normal_files(self, tmp_path: Path) -> None:
        """Test is_restricted_path allows normal files."""
        generator = ProjectGenerator(tmp_path)

        # Normal files should not be restricted
        config_file = tmp_path / "config.yml"
        assert generator.is_restricted_path(config_file) is False

        domain_file = tmp_path / "domain.yml"
        assert generator.is_restricted_path(domain_file) is False

        # Files in subdirectories should not be restricted
        data_file = tmp_path / "data" / "nlu.yml"
        assert generator.is_restricted_path(data_file) is False

    def test_bot_file_paths_excludes_restricted_paths(self, tmp_path: Path) -> None:
        """Test bot_file_paths only returns non-restricted file paths."""
        generator = ProjectGenerator(tmp_path)

        # Create various types of files
        (tmp_path / "config.yml").write_text("version: '3.1'")
        (tmp_path / "domain.yml").write_text("version: '3.1'")
        (tmp_path / ".hidden_file").write_text("hidden")

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model")

        rasa_dir = tmp_path / ".rasa"
        rasa_dir.mkdir()
        (rasa_dir / "cache").write_text("cache")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "nlu.yml").write_text("nlu data")

        # Get bot file paths
        paths = list(generator.bot_file_paths())
        path_names = [p.name for p in paths]

        # Should include normal files
        assert "config.yml" in path_names
        assert "domain.yml" in path_names
        assert "nlu.yml" in path_names

        # Should exclude restricted files
        assert ".hidden_file" not in path_names
        assert "model.tar.gz" not in path_names
        assert "cache" not in path_names

    def test_ensure_all_files_are_writable_allows_normal_files(
        self, tmp_path: Path
    ) -> None:
        """Test ensure_all_files_are_writable allows normal files."""
        generator = ProjectGenerator(tmp_path)

        files = {
            "config.yml": "version: '3.1'",
            "domain.yml": "version: '3.1'",
            "data/nlu.yml": "nlu data",
        }

        # Should not raise any exception
        generator.ensure_all_files_are_writable(files)

    def test_ensure_all_files_are_writable_rejects_restricted_files(
        self, tmp_path: Path
    ) -> None:
        """Test ensure_all_files_are_writable rejects restricted files."""
        generator = ProjectGenerator(tmp_path)

        # Test .rasa files
        files_with_rasa = {"config.yml": "version: '3.1'", ".rasa/cache": "cache data"}

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.ensure_all_files_are_writable(files_with_rasa)

        # Test models files
        files_with_models = {
            "config.yml": "version: '3.1'",
            "models/model.tar.gz": "model data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.ensure_all_files_are_writable(files_with_models)

        # Test hidden files
        files_with_hidden = {
            "config.yml": "version: '3.1'",
            ".hidden_file": "hidden data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.ensure_all_files_are_writable(files_with_hidden)

    def test_replace_all_bot_files_writes_new_files(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files writes new files correctly."""
        generator = ProjectGenerator(tmp_path)

        files = {
            "config.yml": "version: '3.1'\npipeline: []",
            "domain.yml": "version: '3.1'\nintents: []",
            "data/nlu.yml": "version: '3.1'\nnlu: []",
        }

        generator.replace_all_bot_files(files)

        # Verify files were written
        assert (tmp_path / "config.yml").read_text() == "version: '3.1'\npipeline: []"
        assert (tmp_path / "domain.yml").read_text() == "version: '3.1'\nintents: []"
        assert (tmp_path / "data" / "nlu.yml").read_text() == "version: '3.1'\nnlu: []"

    def test_replace_all_bot_files_deletes_existing_files(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files deletes files not in the request."""
        generator = ProjectGenerator(tmp_path)

        # Create some existing files
        (tmp_path / "config.yml").write_text("old config")
        (tmp_path / "domain.yml").write_text("old domain")
        (tmp_path / "old_file.txt").write_text("old file")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "old_nlu.yml").write_text("old nlu")

        # Replace with new files (excluding old_file.txt and old_nlu.yml)
        files = {
            "config.yml": "new config",
            "domain.yml": "new domain",
            "data/new_nlu.yml": "new nlu",
        }

        generator.replace_all_bot_files(files)

        # Verify new files exist with correct content
        assert (tmp_path / "config.yml").read_text() == "new config"
        assert (tmp_path / "domain.yml").read_text() == "new domain"
        assert (tmp_path / "data" / "new_nlu.yml").read_text() == "new nlu"

        # Verify old files were deleted
        assert not (tmp_path / "old_file.txt").exists()
        assert not (data_dir / "old_nlu.yml").exists()

    def test_replace_all_bot_files_preserves_restricted_files(
        self, tmp_path: Path
    ) -> None:
        """Test replace_all_bot_files preserves .rasa and models directories."""
        generator = ProjectGenerator(tmp_path)

        # Create restricted files
        rasa_dir = tmp_path / ".rasa"
        rasa_dir.mkdir()
        (rasa_dir / "cache").write_text("cache data")

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")

        # Create regular file
        (tmp_path / "config.yml").write_text("old config")

        # Replace with new files
        files = {"config.yml": "new config", "domain.yml": "new domain"}

        generator.replace_all_bot_files(files)

        # Verify restricted files still exist
        assert (rasa_dir / "cache").exists()
        assert (rasa_dir / "cache").read_text() == "cache data"
        assert (models_dir / "model.tar.gz").exists()
        assert (models_dir / "model.tar.gz").read_text() == "model data"

        # Verify new files were written
        assert (tmp_path / "config.yml").read_text() == "new config"
        assert (tmp_path / "domain.yml").read_text() == "new domain"

    def test_replace_all_bot_files_skips_none_content(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files skips files with None content."""
        generator = ProjectGenerator(tmp_path)

        files = {
            "config.yml": "version: '3.1'",
            "domain.yml": None,  # Should be skipped
            "data/nlu.yml": "nlu data",
        }

        generator.replace_all_bot_files(files)

        # Verify only non-None files were written
        assert (tmp_path / "config.yml").exists()
        assert not (tmp_path / "domain.yml").exists()
        assert (tmp_path / "data" / "nlu.yml").exists()

    def test_replace_all_bot_files_rejects_restricted_files(
        self, tmp_path: Path
    ) -> None:
        """Test replace_all_bot_files rejects files in restricted paths."""
        generator = ProjectGenerator(tmp_path)

        files_with_restricted = {
            "config.yml": "version: '3.1'",
            ".rasa/cache": "cache data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.replace_all_bot_files(files_with_restricted)

    def test_replace_all_bot_files_creates_directories(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files creates necessary directories."""
        generator = ProjectGenerator(tmp_path)

        files = {
            "data/flows/greeting.yml": "flow data",
            "data/rules/rules.yml": "rules data",
            "actions/custom_actions.py": "action code",
        }

        generator.replace_all_bot_files(files)

        # Verify directories were created and files written
        assert (tmp_path / "data" / "flows" / "greeting.yml").read_text() == "flow data"
        assert (tmp_path / "data" / "rules" / "rules.yml").read_text() == "rules data"
        assert (tmp_path / "actions" / "custom_actions.py").read_text() == "action code"

    def test_cleanup_empty_directories_removes_empty_dirs(self, tmp_path: Path) -> None:
        """Test _cleanup_empty_directories removes empty directories."""
        generator = ProjectGenerator(tmp_path)

        # Create directory structure
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        nested_empty_dir = tmp_path / "parent" / "empty_child"
        nested_empty_dir.mkdir(parents=True)

        # Create directory with file (should not be removed)
        dir_with_file = tmp_path / "dir_with_file"
        dir_with_file.mkdir()
        (dir_with_file / "file.txt").write_text("content")

        generator._cleanup_empty_directories()

        # Verify empty directories were removed
        assert not empty_dir.exists()
        assert not nested_empty_dir.exists()
        assert not (tmp_path / "parent").exists()  # Parent should also be removed

        # Verify directory with file still exists
        assert dir_with_file.exists()
        assert (dir_with_file / "file.txt").exists()

    def test_cleanup_empty_directories_preserves_restricted_dirs(
        self, tmp_path: Path
    ) -> None:
        """Test _cleanup_empty_directories preserves restricted directories."""
        generator = ProjectGenerator(tmp_path)

        # Create empty restricted directories
        rasa_dir = tmp_path / ".rasa"
        rasa_dir.mkdir()

        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Create empty subdirectory in restricted directory
        rasa_subdir = rasa_dir / "subdir"
        rasa_subdir.mkdir()

        generator._cleanup_empty_directories()

        # Verify restricted directories were preserved
        assert rasa_dir.exists()
        assert models_dir.exists()
        assert rasa_subdir.exists()

    def test_get_bot_files_uses_bot_file_paths(self, tmp_path: Path) -> None:
        """Test get_bot_files uses bot_file_paths to filter files."""
        generator = ProjectGenerator(tmp_path)

        # Create various files
        (tmp_path / "config.yml").write_text("config")
        (tmp_path / "domain.yml").write_text("domain")
        (tmp_path / ".hidden_file").write_text("hidden")

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model")

        bot_files = generator.get_bot_files()

        # Should include normal files
        assert "config.yml" in bot_files
        assert "domain.yml" in bot_files
        assert bot_files["config.yml"] == "config"
        assert bot_files["domain.yml"] == "domain"

        # Should exclude restricted files
        assert ".hidden_file" not in bot_files
        assert "models/model.tar.gz" not in bot_files
