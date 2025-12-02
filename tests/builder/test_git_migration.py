import asyncio
from pathlib import Path
from typing import Dict, Optional

import pytest

from rasa.builder.git_service import GitService
from rasa.builder.project_generator.project_generator import ProjectGenerator


class TestGitMigration:
    """Tests for automatic Git migration and defensive initialization."""

    def test_migration_initializes_repo_and_creates_initial_commit(
        self, tmp_path: Path
    ) -> None:
        """Existing projects without .git are migrated on ProjectGenerator init."""
        # Arrange: create an existing project with files but no .git
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "config.yml").write_text("version: '3.1'\npipeline: []")
        (project_dir / "domain.yml").write_text("version: '3.1'\nintents: []")

        # Act: initialize ProjectGenerator which should migrate
        generator = ProjectGenerator(str(project_dir))

        # Assert: .git exists and there is an initial commit
        assert (project_dir / ".git").exists()
        sha = asyncio.run(generator.git_service.get_current_commit_sha())
        assert isinstance(sha, str)
        assert sha != ""

    def test_migration_is_noop_when_repo_already_exists(self, tmp_path: Path) -> None:
        """If .git already exists, migration should not change HEAD."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "config.yml").write_text("version: '3.1'\npipeline: []")

        # Manually initialize git repo and commit
        git = GitService(str(project_dir))
        git._initialize_git_repository()
        git._setup_git_configuration()
        git._create_gitignore()
        git.run_git_command_sync(["add", "."])
        git.run_git_command_sync(
            [
                "commit",
                "-m",
                "Initial manual commit",
            ]
        )
        sha_before = git.run_git_command_sync(["rev-parse", "HEAD"], check_output=True)
        assert sha_before is not None
        sha_before = sha_before.strip()

        # Act: construct ProjectGenerator (should not add new commit)
        _ = ProjectGenerator(str(project_dir))

        # Assert: HEAD remains the same
        sha_after = git.run_git_command_sync(["rev-parse", "HEAD"], check_output=True)
        assert sha_after is not None
        assert sha_after.strip() == sha_before

    @pytest.mark.asyncio
    async def test_commit_guard_initializes_repo_for_empty_project(
        self, tmp_path: Path
    ) -> None:
        """Empty projects get a repo lazily when first write/commit happens."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # No files yet; ProjectGenerator should not eagerly create .git
        generator = ProjectGenerator(str(project_dir))
        assert not (project_dir / ".git").exists()

        # Perform a write path that triggers _commit_changes (defensive ensure)
        files: Dict[str, Optional[str]] = {
            "config.yml": "version: '3.1'\npipeline: []",
            "domain.yml": "version: '3.1'\nintents: []",
        }

        await generator.update_bot_files(files)

        # Repo should now exist with a commit
        assert (project_dir / ".git").exists()
        sha = await generator.git_service.get_current_commit_sha()
        assert isinstance(sha, str)
        assert sha != ""

    def test_migration_creates_gitignore(self, tmp_path: Path) -> None:
        """Migration writes a .gitignore including models/ and .rasa/."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "config.yml").write_text("version: '3.1'")

        _ = ProjectGenerator(str(project_dir))

        gitignore = project_dir / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()
        # Check for key ignores from GitService._create_gitignore
        assert "models/" in content
        assert "__pycache__/" in content
        assert ".rasa/" in content
