import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from rasa.builder.copilot.constants import PROMPT_TO_BOT_KEY, PROMPT_TO_BOT_TEMPLATE_KEY
from rasa.builder.copilot.copilot_templated_message_provider import (
    copilot_welcome_messages,
)
from rasa.builder.exceptions import ProjectGenerationError, ValidationError
from rasa.builder.models import GitCommitInfo
from rasa.builder.project_generator.project_generator import (
    ProjectGenerator,
    is_restricted_path,
)
from rasa.cli.scaffold import ProjectTemplateName
from rasa.utils.io import InvalidPathException


class TestProjectGenerator:
    """Test ProjectGenerator class methods."""

    def test_is_empty_with_empty_directory(self, tmp_path: Path) -> None:
        """Test is_empty returns True for empty directory."""
        generator = ProjectGenerator(str(tmp_path))
        assert generator.is_empty() is True

    def test_is_empty_with_files(self, tmp_path: Path) -> None:
        """Test is_empty returns False when directory contains files."""
        (tmp_path / "config.yml").write_text("version: '3.1'")
        generator = ProjectGenerator(str(tmp_path))
        assert generator.is_empty() is False

    def test_is_empty_ignores_hidden_files(self, tmp_path: Path) -> None:
        """Test is_empty ignores hidden files and directories."""
        (tmp_path / ".hidden_file").write_text("hidden")
        (tmp_path / ".hidden_dir").mkdir()
        generator = ProjectGenerator(str(tmp_path))
        assert generator.is_empty() is True

    def test_is_empty_with_empty_subdirectories(self, tmp_path: Path) -> None:
        """Test is_empty returns True when directory contains only empty subdirs."""
        (tmp_path / "data").mkdir()
        generator = ProjectGenerator(str(tmp_path))
        # The implementation only checks for files, not directories
        assert generator.is_empty() is True

    def test_is_empty_with_files_in_subdirectories(self, tmp_path: Path) -> None:
        """Test is_empty returns True when files are only in subdirs."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "file.txt").write_text("content")
        generator = ProjectGenerator(str(tmp_path))
        # Should return True because is_empty only checks for files at the root level
        assert generator.is_empty() is True

    def test_is_restricted_path_hidden_files(self, tmp_path: Path) -> None:
        """Test is_restricted_path correctly identifies hidden files."""

        # Hidden files should be restricted
        hidden_file = tmp_path / ".hidden_file"
        assert is_restricted_path(tmp_path, hidden_file) is True

        # Hidden directories should be restricted
        hidden_dir = tmp_path / ".hidden_dir"
        assert is_restricted_path(tmp_path, hidden_dir) is True

        # .rasa directory should be restricted
        rasa_dir = tmp_path / ".rasa"
        assert is_restricted_path(tmp_path, rasa_dir) is True

    def test_is_restricted_path_models_directory(self, tmp_path: Path) -> None:
        """Test is_restricted_path correctly identifies models directory."""

        # Models directory should be restricted
        models_dir = tmp_path / "models"
        assert is_restricted_path(tmp_path, models_dir) is True

        # Files in models directory should be restricted
        models_file = tmp_path / "models" / "model.tar.gz"
        assert is_restricted_path(tmp_path, models_file) is True

    def test_is_restricted_path_pycache_directory(self, tmp_path: Path) -> None:
        """Test is_restricted_path correctly identifies __pycache__ directory."""

        # __pycache__ directory should be restricted
        pycache_dir = tmp_path / "actions" / "__pycache__"
        assert is_restricted_path(tmp_path, pycache_dir) is True

    def test_is_restricted_path_normal_files(self, tmp_path: Path) -> None:
        """Test is_restricted_path allows normal files."""

        # Normal files should not be restricted
        config_file = tmp_path / "config.yml"
        assert is_restricted_path(tmp_path, config_file) is False

        domain_file = tmp_path / "domain.yml"
        assert is_restricted_path(tmp_path, domain_file) is False

        # Files in subdirectories should not be restricted
        data_file = tmp_path / "data" / "nlu.yml"
        assert is_restricted_path(tmp_path, data_file) is False

    def test_bot_file_paths_excludes_restricted_paths(self, tmp_path: Path) -> None:
        """Test bot_file_paths only returns non-restricted file paths."""
        generator = ProjectGenerator(str(tmp_path))

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
        generator = ProjectGenerator(str(tmp_path))

        files: dict[str, str | None] = {
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
        generator = ProjectGenerator(str(tmp_path))

        # Test .rasa files
        files_with_rasa: dict[str, str | None] = {
            "config.yml": "version: '3.1'",
            ".rasa/cache": "cache data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.ensure_all_files_are_writable(files_with_rasa)

        # Test models files
        files_with_models: dict[str, str | None] = {
            "config.yml": "version: '3.1'",
            "models/model.tar.gz": "model data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.ensure_all_files_are_writable(files_with_models)

        # Test hidden files
        files_with_hidden: dict[str, str | None] = {
            "config.yml": "version: '3.1'",
            ".hidden_file": "hidden data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            generator.ensure_all_files_are_writable(files_with_hidden)

    def test_replace_all_bot_files_writes_new_files(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files writes new files correctly."""
        generator = ProjectGenerator(str(tmp_path))

        files: dict[str, str | None] = {
            "config.yml": "version: '3.1'\npipeline: []",
            "domain.yml": "version: '3.1'\nintents: []",
            "data/nlu.yml": "version: '3.1'\nnlu: []",
        }

        # Mock the commit method since we're testing file operations, not git
        with patch.object(
            generator,
            "unsafe_commit_changes",
            new_callable=AsyncMock,
            return_value="mock_sha",
        ):
            commit_info = GitCommitInfo(
                message="Test commit", author="test_user", email="test@example.com"
            )
            asyncio.run(generator.replace_all_bot_files(files, commit_info))

        # Verify files were written
        assert (tmp_path / "config.yml").read_text() == "version: '3.1'\npipeline: []"
        assert (tmp_path / "domain.yml").read_text() == "version: '3.1'\nintents: []"
        assert (tmp_path / "data" / "nlu.yml").read_text() == "version: '3.1'\nnlu: []"

    def test_replace_all_bot_files_deletes_existing_files(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files deletes files not in the request."""
        generator = ProjectGenerator(str(tmp_path))

        # Create some existing files
        (tmp_path / "config.yml").write_text("old config")
        (tmp_path / "domain.yml").write_text("old domain")
        (tmp_path / "old_file.txt").write_text("old file")

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "old_nlu.yml").write_text("old nlu")

        # Replace with new files (excluding old_file.txt and old_nlu.yml)
        files: dict[str, str | None] = {
            "config.yml": "new config",
            "domain.yml": "new domain",
            "data/new_nlu.yml": "new nlu",
        }

        # Mock the commit method since we're testing file operations, not git
        with patch.object(
            generator,
            "unsafe_commit_changes",
            new_callable=AsyncMock,
            return_value="mock_sha",
        ):
            commit_info = GitCommitInfo(
                message="Test commit", author="test_user", email="test@example.com"
            )
            asyncio.run(generator.replace_all_bot_files(files, commit_info))

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
        generator = ProjectGenerator(str(tmp_path))

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
        files: dict[str, str | None] = {
            "config.yml": "new config",
            "domain.yml": "new domain",
        }

        # Mock the commit method since we're testing file operations, not git
        with patch.object(
            generator,
            "unsafe_commit_changes",
            new_callable=AsyncMock,
            return_value="mock_sha",
        ):
            commit_info = GitCommitInfo(
                message="Test commit", author="test_user", email="test@example.com"
            )
            asyncio.run(generator.replace_all_bot_files(files, commit_info))

        # Verify restricted files still exist
        assert (rasa_dir / "cache").exists()
        assert (rasa_dir / "cache").read_text() == "cache data"
        assert (models_dir / "model.tar.gz").exists()
        assert (models_dir / "model.tar.gz").read_text() == "model data"

        # Verify new files were written
        assert (tmp_path / "config.yml").read_text() == "new config"
        assert (tmp_path / "domain.yml").read_text() == "new domain"

    def test_replace_all_bot_files_dumps_empty_files(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files dumps empty files."""
        generator = ProjectGenerator(str(tmp_path))

        files = {
            "config.yml": "version: '3.1'",
            "domain.yml": None,  # Should be dumped as an empty file
            "data/nlu.yml": "nlu data",
        }

        # Mock the commit method since we're testing file operations, not git
        with patch.object(
            generator,
            "unsafe_commit_changes",
            new_callable=AsyncMock,
            return_value="mock_sha",
        ):
            commit_info = GitCommitInfo(
                message="Test commit", author="test_user", email="test@example.com"
            )
            asyncio.run(generator.replace_all_bot_files(files, commit_info))

        # Verify empty files were written
        assert (tmp_path / "config.yml").exists()
        assert (tmp_path / "domain.yml").read_text() == ""
        assert (tmp_path / "data" / "nlu.yml").exists()

    def test_replace_all_bot_files_rejects_restricted_files(
        self, tmp_path: Path
    ) -> None:
        """Test replace_all_bot_files rejects files in restricted paths."""
        generator = ProjectGenerator(str(tmp_path))

        files_with_restricted: dict[str, str | None] = {
            "config.yml": "version: '3.1'",
            ".rasa/cache": "cache data",
        }

        with pytest.raises(InvalidPathException, match="restricted from editing"):
            # Mock the commit method since we're testing file operations, not git
            with patch.object(
                generator,
                "unsafe_commit_changes",
                new_callable=AsyncMock,
                return_value="mock_sha",
            ):
                commit_info = GitCommitInfo(
                    message="Test commit", author="test_user", email="test@example.com"
                )
                asyncio.run(
                    generator.replace_all_bot_files(files_with_restricted, commit_info)
                )

    def test_replace_all_bot_files_creates_directories(self, tmp_path: Path) -> None:
        """Test replace_all_bot_files creates necessary directories."""
        generator = ProjectGenerator(str(tmp_path))

        files: dict[str, str | None] = {
            "data/flows/greeting.yml": "flow data",
            "data/rules/rules.yml": "rules data",
            "actions/custom_actions.py": "action code",
        }

        # Mock the commit method since we're testing file operations, not git
        with patch.object(
            generator,
            "unsafe_commit_changes",
            new_callable=AsyncMock,
            return_value="mock_sha",
        ):
            commit_info = GitCommitInfo(
                message="Test commit", author="test_user", email="test@example.com"
            )
            asyncio.run(generator.replace_all_bot_files(files, commit_info))

        # Verify directories were created and files written
        assert (tmp_path / "data" / "flows" / "greeting.yml").read_text() == "flow data"
        assert (tmp_path / "data" / "rules" / "rules.yml").read_text() == "rules data"
        assert (tmp_path / "actions" / "custom_actions.py").read_text() == "action code"

    def test_cleanup_empty_directories_removes_empty_dirs(self, tmp_path: Path) -> None:
        """Test _cleanup_empty_directories removes empty directories."""
        generator = ProjectGenerator(str(tmp_path))

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
        generator = ProjectGenerator(str(tmp_path))

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
        generator = ProjectGenerator(str(tmp_path))

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

    def test_get_bot_files_excludes_models_by_default(self, tmp_path: Path) -> None:
        # Create test files
        (tmp_path / "config.yml").write_text("version: '3.1'")
        (tmp_path / "domain.yml").write_text("version: '3.1'")

        # Create models directory with files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")
        (models_dir / "another_model.tar.gz").write_text("another model")

        generator = ProjectGenerator(str(tmp_path))
        bot_files = generator.get_bot_files()

        # Should include config and domain but not models
        assert "config.yml" in bot_files
        assert "domain.yml" in bot_files
        assert "models/model.tar.gz" not in bot_files
        assert "models/another_model.tar.gz" not in bot_files

    def test_get_bot_files_excludes_models_explicitly_true(
        self, tmp_path: Path
    ) -> None:
        # Create test files
        (tmp_path / "config.yml").write_text("version: '3.1'")

        # Create models directory with files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")

        generator = ProjectGenerator(str(tmp_path))
        bot_files = generator.get_bot_files(exclude_models_directory=True)

        # Should exclude models
        assert "config.yml" in bot_files
        assert "models/model.tar.gz" not in bot_files

    def test_get_bot_files_includes_models_when_false(self, tmp_path: Path) -> None:
        # Create test files
        (tmp_path / "config.yml").write_text("version: '3.1'")

        # Create models directory with files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")
        subdir = models_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_model.tar.gz").write_text("nested model")

        generator = ProjectGenerator(str(tmp_path))
        bot_files = generator.get_bot_files(exclude_models_directory=False)

        # Should include both config and models
        assert "config.yml" in bot_files
        assert "models/model.tar.gz" in bot_files
        assert "models/subdir/nested_model.tar.gz" in bot_files

    def test_get_bot_files_still_excludes_hidden_files(self, tmp_path: Path) -> None:
        """Test get_bot_files still excludes hidden files even when including models."""
        # Create test files including hidden ones
        (tmp_path / "config.yml").write_text("version: '3.1'")
        (tmp_path / ".hidden_file").write_text("hidden")

        # Create models directory with hidden files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")
        (models_dir / ".hidden_model").write_text("hidden model")

        # Create hidden directory in models
        hidden_dir = models_dir / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("file in hidden dir")

        generator = ProjectGenerator(str(tmp_path))
        bot_files = generator.get_bot_files(exclude_models_directory=False)

        # Should include visible files but exclude hidden ones
        assert "config.yml" in bot_files
        assert "models/model.tar.gz" in bot_files
        assert ".hidden_file" not in bot_files
        assert "models/.hidden_model" not in bot_files
        assert "models/.hidden_dir/file.txt" not in bot_files

    def test_get_bot_files_excludes_pycache(self, tmp_path: Path) -> None:
        """Test get_bot_files excludes __pycache__ directories."""
        # Create test files
        (tmp_path / "config.yml").write_text("version: '3.1'")

        # Create __pycache__ directory
        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "module.pyc").write_text("compiled python")

        # Create models directory with __pycache__
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")
        models_pycache = models_dir / "__pycache__"
        models_pycache.mkdir()
        (models_pycache / "model_module.pyc").write_text("compiled model")

        generator = ProjectGenerator(str(tmp_path))

        # Test with models excluded
        bot_files_no_models = generator.get_bot_files(exclude_models_directory=True)
        assert "config.yml" in bot_files_no_models
        assert "__pycache__/module.pyc" not in bot_files_no_models
        assert "models/model.tar.gz" not in bot_files_no_models

        # Test with models included
        bot_files_with_models = generator.get_bot_files(exclude_models_directory=False)
        assert "config.yml" in bot_files_with_models
        assert "models/model.tar.gz" in bot_files_with_models
        assert "__pycache__/module.pyc" not in bot_files_with_models
        assert "models/__pycache__/model_module.pyc" not in bot_files_with_models

    def test_get_bot_files_works_with_no_models_directory(self, tmp_path: Path) -> None:
        """Test get_bot_files works when no models directory exists."""
        # Create test files without models directory
        (tmp_path / "config.yml").write_text("version: '3.1'")
        (tmp_path / "domain.yml").write_text("version: '3.1'")

        generator = ProjectGenerator(str(tmp_path))

        # Both settings should work fine
        bot_files_exclude = generator.get_bot_files(exclude_models_directory=True)
        bot_files_include = generator.get_bot_files(exclude_models_directory=False)

        # Should have same files in both cases
        assert bot_files_exclude == bot_files_include
        assert "config.yml" in bot_files_exclude
        assert "domain.yml" in bot_files_exclude

    def test_get_bot_files_models_directory_not_at_root(self, tmp_path: Path) -> None:
        """Test get_bot_files only excludes models directory at project root."""
        # Create test files
        (tmp_path / "config.yml").write_text("version: '3.1'")

        # Create models directory at root
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model.tar.gz").write_text("model data")

        # Create a subdirectory also named "models" (should not be excluded)
        subdir = tmp_path / "data"
        subdir.mkdir()
        sub_models = subdir / "models"
        sub_models.mkdir()
        (sub_models / "data_model.json").write_text("data model")

        generator = ProjectGenerator(str(tmp_path))
        bot_files = generator.get_bot_files(exclude_models_directory=True)

        # Should exclude root models but include nested models directory
        assert "config.yml" in bot_files
        assert "models/model.tar.gz" not in bot_files  # Root models excluded
        assert "data/models/data_model.json" in bot_files  # Nested models included

    @pytest.mark.asyncio
    async def test_attempt_generation_success(self, tmp_path: Path) -> None:
        """Test successful project generation attempt."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock dependencies
        with (
            patch.object(
                generator, "generate_response", new_callable=AsyncMock
            ) as mock_generate,
            patch.object(
                generator, "_update_bot_files_from_llm_response", new_callable=AsyncMock
            ) as mock_update,
            patch.object(
                generator, "_validate_generated_project", new_callable=AsyncMock
            ) as mock_validate,
            patch.object(
                generator, "get_bot_files", return_value={"config.yml": "test"}
            ),
        ):
            mock_generate.return_value = {"domain": {}, "flows": {"flows": {}}}
            mock_update.return_value = "commit_sha_123"
            mock_validate.return_value = None

            # Execute
            commit_sha = await generator._attempt_generation(
                initial_messages=[{"role": "system", "content": "test"}],
                error_feedback_messages=[],
                attempts_left=3,
                max_retries=3,
            )

            # Verify
            assert commit_sha == "commit_sha_123"
            mock_generate.assert_called_once()
            mock_update.assert_called_once()
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_attempt_generation_validation_error(self, tmp_path: Path) -> None:
        """Test generation attempt with validation error."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock dependencies
        validation_error = ValidationError("Validation failed")
        with (
            patch.object(
                generator, "generate_response", new_callable=AsyncMock
            ) as mock_generate,
            patch.object(
                generator, "_update_bot_files_from_llm_response", new_callable=AsyncMock
            ) as mock_update,
            patch.object(
                generator, "_validate_generated_project", new_callable=AsyncMock
            ) as mock_validate,
            patch.object(
                generator, "get_bot_files", return_value={"config.yml": "test"}
            ),
        ):
            mock_generate.return_value = {"domain": {}, "flows": {"flows": {}}}
            mock_update.return_value = "commit_sha_123"
            mock_validate.side_effect = validation_error

            # Execute and verify exception is raised
            with pytest.raises(ValidationError):
                await generator._attempt_generation(
                    initial_messages=[{"role": "system", "content": "test"}],
                    error_feedback_messages=[],
                    attempts_left=3,
                    max_retries=3,
                )

            # Verify methods were called
            mock_generate.assert_called_once()
            mock_update.assert_called_once()
            mock_validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_attempt_generation_generic_error(self, tmp_path: Path) -> None:
        """Test generation attempt with generic error."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock generate_response to raise an exception
        with patch.object(
            generator, "generate_response", new_callable=AsyncMock
        ) as mock_generate:
            mock_generate.side_effect = Exception("LLM error")

            # Execute and verify exception is raised
            with pytest.raises(Exception, match="LLM error"):
                await generator._attempt_generation(
                    initial_messages=[{"role": "system", "content": "test"}],
                    error_feedback_messages=[],
                    attempts_left=3,
                    max_retries=3,
                )

    @pytest.mark.asyncio
    async def test_generate_project_with_retries_success_first_attempt(
        self, tmp_path: Path
    ) -> None:
        """Test successful project generation on first attempt."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock dependencies
        with (
            patch.object(
                generator, "init_from_template", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                generator, "_attempt_generation", new_callable=AsyncMock
            ) as mock_attempt,
            patch.object(
                generator,
                "_get_bot_data_for_llm",
                return_value={"domain": {}, "flows": {}},
            ),
            patch.object(
                generator,
                "_create_system_message",
                return_value={"role": "system", "content": "test"},
            ),
            patch.object(
                generator,
                "_create_user_request_message",
                return_value={"role": "user", "content": "test"},
            ),
        ):
            mock_init.return_value = "init_commit_sha"
            mock_attempt.return_value = "final_commit_sha"

            # Execute
            attempts, commit_sha = await generator.generate_project_with_retries(
                skill_description="Build a banking bot",
                template=ProjectTemplateName.BASIC,
                max_retries=3,
            )

            # Verify
            assert attempts == 0  # Success on first attempt
            assert commit_sha == "final_commit_sha"
            mock_init.assert_called_once_with(ProjectTemplateName.BASIC)
            mock_attempt.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_project_with_retries_success_after_retry(
        self, tmp_path: Path
    ) -> None:
        """Test successful project generation after validation errors."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock dependencies
        validation_error = ValidationError("Invalid domain")
        with (
            patch.object(
                generator, "init_from_template", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                generator, "_attempt_generation", new_callable=AsyncMock
            ) as mock_attempt,
            patch.object(
                generator,
                "_get_bot_data_for_llm",
                return_value={"domain": {}, "flows": {}},
            ),
            patch.object(
                generator,
                "_create_system_message",
                return_value={"role": "system", "content": "test"},
            ),
            patch.object(
                generator,
                "_create_user_request_message",
                return_value={"role": "user", "content": "test"},
            ),
            patch.object(
                generator,
                "_create_error_feedback_messages",
                return_value=[{"role": "user", "content": "error feedback"}],
            ),
            patch.object(
                generator, "_get_copilot_error_guidance", new_callable=AsyncMock
            ) as mock_guidance,
        ):
            mock_init.return_value = "init_commit_sha"
            # First attempt fails, second succeeds
            mock_attempt.side_effect = [validation_error, "final_commit_sha"]
            mock_guidance.return_value = "Fix the error"

            # Execute
            attempts, commit_sha = await generator.generate_project_with_retries(
                skill_description="Build a banking bot",
                template=ProjectTemplateName.BASIC,
                max_retries=3,
            )

            # Verify
            assert attempts == 1  # Success on second attempt
            assert commit_sha == "final_commit_sha"
            assert mock_attempt.call_count == 2
            mock_guidance.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_project_with_retries_exhausts_retries(
        self, tmp_path: Path
    ) -> None:
        """Test project generation exhausting all retries."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock dependencies
        validation_error = ValidationError("Invalid domain")
        with (
            patch.object(
                generator, "init_from_template", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                generator, "_attempt_generation", new_callable=AsyncMock
            ) as mock_attempt,
            patch.object(
                generator,
                "_get_bot_data_for_llm",
                return_value={"domain": {}, "flows": {}},
            ),
            patch.object(
                generator,
                "_create_system_message",
                return_value={"role": "system", "content": "test"},
            ),
            patch.object(
                generator,
                "_create_user_request_message",
                return_value={"role": "user", "content": "test"},
            ),
            patch.object(
                generator,
                "_create_error_feedback_messages",
                return_value=[{"role": "user", "content": "error feedback"}],
            ),
            patch.object(
                generator, "_get_copilot_error_guidance", new_callable=AsyncMock
            ),
        ):
            mock_init.return_value = "init_commit_sha"
            # All attempts fail
            mock_attempt.side_effect = validation_error

            # Execute - should return max_retries and init commit sha
            attempts, commit_sha = await generator.generate_project_with_retries(
                skill_description="Build a banking bot",
                template=ProjectTemplateName.BASIC,
                max_retries=2,
            )

            # Verify
            assert attempts == 2
            assert commit_sha == "init_commit_sha"
            assert mock_attempt.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_project_with_retries_generic_error(
        self, tmp_path: Path
    ) -> None:
        """Test project generation with generic error."""
        generator = ProjectGenerator(str(tmp_path))

        # Mock dependencies
        with (
            patch.object(
                generator, "init_from_template", new_callable=AsyncMock
            ) as mock_init,
            patch.object(
                generator, "_attempt_generation", new_callable=AsyncMock
            ) as mock_attempt,
            patch.object(
                generator,
                "_get_bot_data_for_llm",
                return_value={"domain": {}, "flows": {}},
            ),
            patch.object(
                generator,
                "_create_system_message",
                return_value={"role": "system", "content": "test"},
            ),
            patch.object(
                generator,
                "_create_user_request_message",
                return_value={"role": "user", "content": "test"},
            ),
        ):
            mock_init.return_value = "init_commit_sha"
            # All attempts fail with generic error
            mock_attempt.side_effect = Exception("LLM error")

            # Execute and verify exception is raised
            with pytest.raises(
                ProjectGenerationError, match="Failed to generate Rasa project"
            ):
                await generator.generate_project_with_retries(
                    skill_description="Build a banking bot",
                    template=ProjectTemplateName.BASIC,
                    max_retries=2,
                )

            # Verify attempts were made
            assert mock_attempt.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_commit_message(self, tmp_path: Path) -> None:
        """Test commit message generation."""
        generator = ProjectGenerator(str(tmp_path))

        diff_output = "M\tconfig.yml\nA\tdomain.yml"
        detailed_diff = "diff --git a/config.yml b/config.yml\n+version: 3.1"
        llm_response = "Update config and add domain"

        # Create a mock ChatCompletion response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = llm_response

        with (
            patch.object(
                generator.git_service,
                "run_git_command",
                new_callable=AsyncMock,
            ) as mock_git,
            patch.object(
                generator,
                "_generate_text",
                new_callable=AsyncMock,
            ) as mock_llm,
            patch(
                "rasa.builder.project_generator.project_generator.CommitMessageGenerationLangfuseTelemetry.update_commit_message_generation_input"
            ) as mock_telemetry_input,
            patch(
                "rasa.builder.project_generator.project_generator.CommitMessageGenerationLangfuseTelemetry.update_commit_message_generation_output"
            ) as mock_telemetry_output,
        ):
            # Mock git commands
            mock_git.side_effect = [diff_output, detailed_diff]
            # Mock LLM response
            mock_llm.return_value = mock_response

            # Execute
            result = await generator._generate_commit_message()

            # Verify
            assert result == "Update config and add domain"
            assert mock_git.call_count == 2
            mock_git.assert_any_call(["diff", "--name-status"], check_output=True)
            mock_git.assert_any_call(["diff", "--unified=2"], check_output=True)
            mock_llm.assert_called_once()
            mock_telemetry_input.assert_called_once()
            mock_telemetry_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_commit_message_no_diff(self, tmp_path: Path) -> None:
        """Test commit message generation."""
        generator = ProjectGenerator(str(tmp_path))

        diff_output = ""

        with (
            patch.object(
                generator.git_service,
                "run_git_command",
                new_callable=AsyncMock,
            ) as mock_git,
        ):
            # Mock git commands
            mock_git.side_effect = [diff_output]

            # Execute
            result = await generator._generate_commit_message()

            # Verify default message is used when no diff
            assert result == "Update files"

    @pytest.mark.asyncio
    async def test_generate_commit_message_too_long_message(
        self, tmp_path: Path
    ) -> None:
        """Test commit message generation."""
        generator = ProjectGenerator(str(tmp_path))

        diff_output = "M\tconfig.yml\nA\tdomain.yml"
        detailed_diff = "diff --git a/config.yml b/config.yml\n+version: 3.1"
        llm_response = "Update all flows, domain data and config > 36"

        # Create a mock ChatCompletion response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = llm_response

        with (
            patch.object(
                generator.git_service,
                "run_git_command",
                new_callable=AsyncMock,
            ) as mock_git,
            patch.object(
                generator,
                "_generate_text",
                new_callable=AsyncMock,
            ) as mock_llm,
        ):
            # Mock git commands
            mock_git.side_effect = [diff_output, detailed_diff]
            # Mock LLM response
            mock_llm.return_value = mock_response

            # Execute
            result = await generator._generate_commit_message()

            # Verify default message when LLM response too long
            assert result == "Update files"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "flows,llm_response,expected_uses_template,should_call_llm",
        [
            pytest.param(
                {
                    "hello_world_greeting": {
                        "steps": [{"action": "utter_hello_world", "next": "END"}],
                        "description": "Responds to user greetings with 'Hello World'.",
                    }
                },
                "- *Say hello to me*",
                True,
                True,
                id="generated_message",
            ),
            pytest.param(
                {},
                None,
                False,
                False,
                id="no_flows",
            ),
            pytest.param(
                {
                    "hello_world_greeting": {
                        "steps": [{"action": "utter_hello_world", "next": "END"}],
                        "description": "Responds to user greetings with 'Hello World'.",
                    }
                },
                None,
                False,
                True,
                id="no_response",
            ),
            pytest.param(
                {
                    "hello_world_greeting": {
                        "steps": [{"action": "utter_hello_world", "next": "END"}],
                        "description": "Responds to user greetings with 'Hello World'.",
                    }
                },
                "",
                False,
                True,
                id="empty_response",
            ),
            pytest.param(
                {
                    "hello_world_greeting": {
                        "steps": [{"action": "utter_hello_world", "next": "END"}],
                        "description": "Responds to user greetings with 'Hello World'.",
                    }
                },
                " *Say hello to me*",
                False,
                True,
                id="wrong_format_response",
            ),
            pytest.param(
                {
                    "greeting_flow": {
                        "steps": [{"action": "utter_greet", "next": "END"}],
                        "description": "Greet the user",
                    },
                    "goodbye_flow": {
                        "steps": [{"action": "utter_goodbye", "next": "END"}],
                        "description": "Say goodbye to the user",
                    },
                    "help_flow": {
                        "steps": [{"action": "utter_help", "next": "END"}],
                        "description": "Provide help information",
                    },
                },
                "- *Hello*\n- *Goodbye*\n- *Help me*",
                True,
                True,
                id="multiple_flows",
            ),
            pytest.param(
                {
                    "greeting_flow": {
                        "steps": [{"action": "utter_greet", "next": "END"}],
                        "description": "Greet the user",
                    },
                    "goodbye_flow": {
                        "steps": [{"action": "utter_goodbye", "next": "END"}],
                        "description": "Say goodbye to the user",
                    },
                    "help_flow": {
                        "steps": [{"action": "utter_help", "next": "END"}],
                        "description": "Provide help information",
                    },
                },
                "- *Say hello to me*\n- *Hello*\n- *Goodbye*\n- *Help me*",
                False,
                True,
                id="too_many_examples",
            ),
        ],
    )
    async def test_generate_welcome_message(
        self,
        tmp_path: Path,
        flows: dict,
        llm_response: str | None,
        expected_uses_template: bool,
        should_call_llm: bool,
    ) -> None:
        """Test welcome message generation"""
        generator = ProjectGenerator(str(tmp_path))

        welcome_messages = copilot_welcome_messages()
        default_welcome_message = welcome_messages.get(PROMPT_TO_BOT_KEY)
        template_welcome_message = welcome_messages.get(PROMPT_TO_BOT_TEMPLATE_KEY)

        assert default_welcome_message is not None
        assert template_welcome_message is not None

        # Create a mock ChatCompletion response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = llm_response

        with (
            patch.object(
                generator,
                "_get_bot_data_for_llm",
                return_value={"domain": {}, "flows": flows},
            ),
            patch.object(
                generator,
                "_generate_text",
                new_callable=AsyncMock,
            ) as mock_llm,
            patch(
                "rasa.builder.project_generator.project_generator.WelcomeMessageGenerationLangfuseTelemetry.update_welcome_message_generation_input"
            ) as mock_telemetry_input,
            patch(
                "rasa.builder.project_generator.project_generator.WelcomeMessageGenerationLangfuseTelemetry.update_welcome_message_generation_output"
            ) as mock_telemetry_output,
        ):
            # Mock LLM response
            mock_llm.return_value = mock_response

            # Execute
            welcome_message = await generator.generate_welcome_message(
                default_welcome_message=default_welcome_message,
                template_welcome_message=template_welcome_message,
            )

            # Verify LLM calls
            if should_call_llm:
                mock_llm.assert_called_once()
                mock_telemetry_input.assert_called_once()
                mock_telemetry_output.assert_called_once()
            else:
                mock_llm.assert_not_called()
                mock_telemetry_input.assert_not_called()
                mock_telemetry_output.assert_not_called()

            # Verify welcome message
            if expected_uses_template:
                assert welcome_message == template_welcome_message.format(
                    example_questions=llm_response
                )
            else:
                assert welcome_message == default_welcome_message

    @pytest.mark.parametrize(
        "response,max_amount,expected",
        [
            pytest.param(
                "- *Hello*",
                3,
                True,
                id="single_valid_bullet_point",
            ),
            pytest.param(
                "- *Hello*\n- *Goodbye*\n- *Help me*",
                3,
                True,
                id="multiple_valid_bullet_points",
            ),
            pytest.param(
                "- *Hello*\n- *Goodbye*",
                3,
                True,
                id="less_than_max_amount",
            ),
            pytest.param(
                "- *Hello*\n- *Goodbye*\n- *Help me*\n- *Extra*",
                3,
                False,
                id="exceeds_max_amount",
            ),
            pytest.param(
                "",
                3,
                False,
                id="empty_string",
            ),
            pytest.param(
                "\n\n\n",
                3,
                False,
                id="only_whitespace",
            ),
            pytest.param(
                "- Hello",
                3,
                False,
                id="missing_asterisks",
            ),
            pytest.param(
                "*Hello*",
                3,
                False,
                id="missing_dash",
            ),
            pytest.param(
                " *Hello*",
                3,
                False,
                id="missing_dash_with_space",
            ),
            pytest.param(
                "- *Hello",
                3,
                False,
                id="missing_closing_asterisk",
            ),
            pytest.param(
                "- Hello*",
                3,
                False,
                id="missing_opening_asterisk",
            ),
            pytest.param(
                "- **Hello**",
                3,
                False,
                id="double_asterisks",
            ),
            pytest.param(
                "- *Hello*\n- Goodbye",
                3,
                False,
                id="mixed_valid_and_invalid",
            ),
            pytest.param(
                "- *Hello*\n\n- *Goodbye*",
                3,
                True,
                id="valid_with_empty_lines",
            ),
            pytest.param(
                "  - *Hello*  \n  - *Goodbye*  ",
                3,
                True,
                id="valid_with_leading_trailing_whitespace",
            ),
            pytest.param(
                "-*Hello*",
                3,
                False,
                id="no_space_after_dash",
            ),
            pytest.param(
                "- *Hello world*",
                3,
                True,
                id="bullet_point_with_spaces",
            ),
            pytest.param(
                "- *Can I transfer money?*",
                3,
                True,
                id="bullet_point_with_question_mark",
            ),
            pytest.param(
                "- *Check my balance!*",
                3,
                True,
                id="bullet_point_with_exclamation",
            ),
            pytest.param(
                "- *Hello, how are you?*",
                3,
                True,
                id="bullet_point_with_comma",
            ),
            pytest.param(
                "- *Hello*\n- *Goodbye*\n- *Help*",
                2,
                False,
                id="exactly_exceeds_max_amount",
            ),
            pytest.param(
                "- *Hello*\n- *Goodbye*",
                2,
                True,
                id="exactly_at_max_amount",
            ),
            pytest.param(
                "- *Hello*",
                1,
                True,
                id="single_bullet_max_one",
            ),
            pytest.param(
                "- *Hello*\n- *Goodbye*",
                1,
                False,
                id="two_bullets_max_one",
            ),
            pytest.param(
                "- *Text with * asterisk inside*",
                3,
                False,
                id="asterisk_inside_text",
            ),
            pytest.param(
                "- *First*\n- *Second*\nNot a bullet",
                3,
                False,
                id="valid_bullets_with_non_bullet_line",
            ),
        ],
    )
    def test_verify_bullet_points(
        self, tmp_path: Path, response: str, max_amount: int, expected: bool
    ) -> None:
        """Test _verify_bullet_points"""
        generator = ProjectGenerator(str(tmp_path))
        result = generator._verify_bullet_points(response, max_amount)
        assert result == expected
