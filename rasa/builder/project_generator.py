"""Service for generating Rasa projects from prompts."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional

import structlog

from rasa.builder import config
from rasa.builder.exceptions import ProjectGenerationError, ValidationError
from rasa.builder.git_service import (
    DEFAULT_COMMIT_INFO,
    GitOperationInProgressError,
    GitService,
)
from rasa.builder.llm_service import get_skill_generation_messages, llm_service
from rasa.builder.logging_utils import capture_exception_with_context
from rasa.builder.models import BotFiles, GitCommitInfo
from rasa.builder.project_info import ProjectInfo, ensure_first_used, load_project_info
from rasa.builder.template_cache import copy_cache_for_template_if_available
from rasa.builder.training_service import TrainingInput
from rasa.builder.validation_service import validate_project
from rasa.cli.scaffold import ProjectTemplateName, create_initial_project
from rasa.shared.constants import DEFAULT_MODELS_PATH
from rasa.shared.core.flows import yaml_flows_io
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string
from rasa.utils.io import InvalidPathException, subpath

structlogger = structlog.get_logger()


class ProjectGenerator:
    """Service for generating Rasa projects from skill descriptions."""

    def __init__(self, project_folder: str) -> None:
        """Initialize the project generator with a folder for file persistence.

        Args:
            project_folder: Path to the folder where project files will be stored
        """
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.git_service = GitService(project_folder)

        # Migration: if existing projects don't have a git repo, initialize one
        # and create an initial snapshot commit synchronously to capture the
        # current state before any writes happen in this process.
        self.migrate_git_repository_if_needed()

    @property
    def project_info(self) -> ProjectInfo:
        """Get the project info."""
        return load_project_info(self.project_folder)

    def is_empty(self) -> bool:
        """Check if the project folder is empty.

        Excluding hidden paths.
        """
        return not any(
            file.is_file()
            for file in self.project_folder.iterdir()
            if not file.name.startswith(".")
        )

    async def init_from_template(self, template: ProjectTemplateName) -> str:
        """Create the initial project files.

        Raises:
            GitOperationInProgressError: If another git operation is in progress
        """
        # Acquire lock for entire operation (cleanup + file creation + commit)
        async with self.git_service.git_operation():
            self.cleanup()
            create_initial_project(self.project_folder.as_posix(), template)
            # If a local cache for this template exists, copy it into the project.
            # We no longer download here to avoid blocking project creation.
            await copy_cache_for_template_if_available(template, self.project_folder)
            # needs to happen after caching, as we download/copy .rasa and that would
            # overwrite the project info file in .rasa
            ensure_first_used(self.project_folder)

            self._ensure_git_repository()

            # Create initial Git commit
            # Note: Use internal commit method since we already hold the lock
            return await self.git_service._commit_changes_internal(
                DEFAULT_COMMIT_INFO.model_copy(
                    update={
                        "message": f"Initialize project from {template.value} template"
                    }
                )
            )

    async def generate_project_with_retries(
        self,
        skill_description: str,
        template: ProjectTemplateName,
        max_retries: Optional[int] = None,
    ) -> str:
        """Generate a Rasa project with retry logic for validation failures.

        Args:
            skill_description: Natural language description of the skill
            rasa_config: Rasa configuration dictionary
            template: Project template to use for the initial project
            max_retries: Maximum number of retry attempts

        Returns:
            Commit sha of the generated files

        Raises:
            ProjectGenerationError: If generation fails after all retries
        """
        if max_retries is None:
            max_retries = config.MAX_RETRIES

        # if ever we do not use the template, we need to ensure first_used is set
        # separately!
        await self.init_from_template(template)

        project_data = self._get_bot_data_for_llm()

        initial_messages = get_skill_generation_messages(
            skill_description, project_data
        )

        async def _generate_with_retry(
            messages: List[Dict[str, Any]], attempts_left: int
        ) -> None:
            try:
                # Generate project data using LLM
                project_data = await llm_service.generate_rasa_project(messages)

                # Update stored bot data
                await self._update_bot_files_from_llm_response(project_data)

                structlogger.info(
                    "project_generator.generated_project",
                    attempts_left=attempts_left,
                )

                # Validate the generated project
                await self._validate_generated_project()

                structlogger.info(
                    "project_generator.validation_success", attempts_left=attempts_left
                )

            except ValidationError as e:
                structlogger.error(
                    "project_generator.validation_error",
                    error=str(e),
                    attempts_left=attempts_left,
                )

                if attempts_left <= 0:
                    raise ProjectGenerationError(
                        f"Failed to generate valid Rasa project: {e}", max_retries
                    )

                # Create error feedback for next attempt
                error_feedback_messages = messages + [
                    {
                        "role": "assistant",
                        "content": json.dumps(project_data),
                    },
                    {
                        "role": "user",
                        "content": dedent(f"""
                            Previous attempt failed validation with error: {e}

                            Please fix the issues and generate a valid Rasa project.
                            Pay special attention to:
                            - Proper YAML syntax
                            - Required fields in domain and flows
                            - Consistent naming between flows and domain
                            - Valid slot types and mappings
                        """).strip(),
                    },
                ]

                await _generate_with_retry(error_feedback_messages, attempts_left - 1)

            except Exception as e:
                structlogger.error(
                    "project_generator.generation_error",
                    error=str(e),
                    attempts_left=attempts_left,
                )

                if attempts_left <= 0:
                    raise ProjectGenerationError(
                        f"Failed to generate Rasa project: {e}", max_retries
                    )

                # For non-validation errors, retry with original messages
                await _generate_with_retry(initial_messages, attempts_left - 1)

        await _generate_with_retry(initial_messages, max_retries)
        return await self.git_service.commit_changes(
            DEFAULT_COMMIT_INFO.model_copy(
                update={"message": "Generated initial project files"}
            )
        )

    async def _validate_generated_project(self) -> None:
        """Validate the generated project using the validation service."""
        importer = self._create_importer()
        validation_error = await validate_project(importer)

        if validation_error:
            raise ValidationError(validation_error)

    def _get_endpoints_file(self) -> Path:
        """Get the endpoints file."""
        return self.project_folder / "endpoints.yml"

    def _get_config_file(self) -> Path:
        """Get the config file."""
        return self.project_folder / "config.yml"

    def get_training_input(self) -> TrainingInput:
        """Get the training input."""
        return TrainingInput(
            importer=self._create_importer(),
            endpoints_file=self._get_endpoints_file(),
            config_file=self._get_config_file(),
        )

    def _create_importer(self) -> TrainingDataImporter:
        """Create a training data importer from the current bot files."""
        try:
            if (self.project_folder / "domain.yml").exists():
                domain_path = self.project_folder / "domain.yml"
            else:
                domain_path = self.project_folder / "domain"

            return TrainingDataImporter.load_from_config(
                config_path=str(self._get_config_file()),
                domain_path=str(domain_path),
                training_data_paths=[str(self.project_folder / "data")],
                args={},
            )

        except Exception as e:
            raise ValidationError(f"Failed to create importer: {e}")

    def get_bot_files(
        self,
        allowed_file_extensions: Optional[List[str]] = None,
        exclude_docs_directory: bool = False,
        exclude_models_directory: bool = True,
    ) -> BotFiles:
        """Get the current bot files by reading from disk.

        Args:
            allowed_file_extensions: Optional list of file extensions to include.
                If None, fetch all files. If provided, only fetch files with matching
                extensions. Use `""` empty string to allow files with no extensions.
            exclude_docs_directory: Optional boolean indicating whether to exclude.
            exclude_models_directory: Optional boolean indicating whether to exclude.

        Returns:
            Dictionary of file contents with relative paths as keys
        """
        bot_files: BotFiles = {}

        for file in self.bot_file_paths(exclude_models_directory):
            relative_path = file.relative_to(self.project_folder)

            # Exclude the docs directory if specified
            if exclude_docs_directory and relative_path.parts[0] == "docs":
                continue

            # Exclude the files by file extensions if specified
            if allowed_file_extensions is not None:
                allowed_file_extensions = [
                    ext.lower() for ext in allowed_file_extensions
                ]
                if file.suffix.lstrip(".").lower() not in allowed_file_extensions:
                    continue
            # Read file content and store with relative path as key
            try:
                bot_files[relative_path.as_posix()] = file.read_text(encoding="utf-8")
            except Exception as e:
                structlogger.debug(
                    "project_generator.get_bot_files.error",
                    error=str(e),
                    file_path=file.as_posix(),
                )
                bot_files[relative_path.as_posix()] = None
        return bot_files

    def is_restricted_path(
        self, path: Path, exclude_models_directory: bool = True
    ) -> bool:
        """Check if the path is restricted.

        These paths are excluded from deletion and editing by the user.
        """
        relative_path = path.relative_to(self.project_folder)

        # Skip hidden files and directories (any path component starting with '.')
        # as well as `__pycache__` folders
        if any(part.startswith(".") for part in relative_path.parts):
            return True

        if "__pycache__" in relative_path.parts:
            return True

        # exclude the project_folder / models folder if specified
        if exclude_models_directory and relative_path.parts[0] == DEFAULT_MODELS_PATH:
            return True

        return False

    def _ensure_git_repository(self) -> None:
        """Ensure the project folder is a Git repository on main branch."""
        # Initialize synchronously to avoid event loop conflicts in __init__
        if self.git_service.git_dir.exists():
            return

        structlogger.info(
            "project_generator.init_git_repository",
            project_folder=self.project_folder.as_posix(),
        )

        # Initialize Git repository synchronously
        self.git_service._initialize_git_repository()
        self.git_service._setup_git_configuration()
        self.git_service._create_gitignore()

        structlogger.info(
            "project_generator.git_repository_initialized",
            project_folder=self.project_folder.as_posix(),
        )

    def migrate_git_repository_if_needed(self) -> None:
        """Initialize Git and create an initial commit for existing projects.

        This migrates pre-existing builder instances that were created before
        Git integration existed. If the project folder contains files and no
        `.git` directory, we initialize a repo and create an initial commit
        capturing the current state. Hidden files and folders like `.rasa/` and
        large artifacts like `models/` are excluded via `.gitignore`.
        """
        try:
            if self.git_service.git_dir.exists():
                return

            # Only migrate if there is something to snapshot. Hidden files are
            # ignored by is_empty().
            if self.is_empty():
                return

            structlogger.info(
                "project_generator.migration_git_repository_start",
                project_folder=self.project_folder.as_posix(),
            )

            # Initialize repo and basic config + .gitignore (sync, safe in __init__)
            self.git_service._initialize_git_repository()
            self.git_service._setup_git_configuration()
            self.git_service._create_gitignore()

            # Stage and create an initial snapshot commit synchronously
            # It's fine if commit fails because nothing is staged.
            self.git_service.run_git_command_sync(["add", "."])
            try:
                self.git_service.run_git_command_sync(
                    [
                        "commit",
                        "-m",
                        "Initialize Git history from existing project state",
                    ]
                )
                structlogger.info(
                    "project_generator.migration_git_repository_completed",
                    project_folder=self.project_folder.as_posix(),
                )
            except subprocess.CalledProcessError as e:
                # No staged changes or commit failed; log and continue
                structlogger.warning(
                    "project_generator.migration_git_repository_commit_failed",
                    error=str(e),
                    project_folder=self.project_folder.as_posix(),
                )
        except Exception as e:
            structlogger.warning(
                "project_generator.migration_git_repository_failed",
                error=str(e),
                project_folder=self.project_folder.as_posix(),
            )

    async def _commit_changes(self, commit_info: GitCommitInfo) -> str:
        """Commit all changes.

        Note: This method assumes the git operation lock is already held by the caller.
        It uses the internal commit implementation to avoid deadlock.

        Args:
            commit_info: info about the commit

        Returns:
            Commit SHA of the created commit

        Raises:
            GitOperationInProgressError: If another git operation is in progress
        """
        try:
            # Ensure git repository exists (defensive in case migration didn't run)
            if not self.git_service.git_dir.exists():
                self._ensure_git_repository()
            # Generate commit message if not provided
            if commit_info.message is None:
                commit_info.message = await self._generate_commit_message()

            # Commit changes using GitService internal method
            # (assumes lock is already held by caller)
            commit_sha = await self.git_service._commit_changes_internal(commit_info)

            return commit_sha

        except GitOperationInProgressError:
            # Re-raise this exception so it can be handled by the caller
            raise
        except Exception as e:
            structlogger.error(
                "project_generator.git_commit_failed",
                error=str(e),
                project_folder=self.project_folder.as_posix(),
            )
            # Don't fail the operation if Git commit fails, return current commit
            return await self.git_service.get_current_commit_sha()

    async def _generate_commit_message(self) -> str:
        """Generate a meaningful commit message using AI based on the changes.

        Returns:
            A descriptive commit message based on the Git diff
        """
        try:
            # Get the diff of staged changes
            diff_output = (
                await self.git_service.run_git_command(
                    ["diff", "--cached", "--name-status"], check_output=True
                )
                or ""
            )

            if not diff_output:
                return "Update bot files"

            # Get a more detailed diff for context (limited to avoid token limits)
            detailed_diff = (
                await self.git_service.run_git_command(
                    ["diff", "--cached", "--unified=2"], check_output=True
                )
                or ""
            )

            # Limit the diff size to avoid token limits
            if detailed_diff and len(detailed_diff) > 2000:
                detailed_diff = detailed_diff[:2000] + "\n... (diff truncated)"

            # Prepare the prompt for the LLM
            prompt = (
                f"Generate a concise, descriptive Git commit message for the "
                f"following changes to a Rasa chatbot project.\n\n"
                f"The commit message should:\n"
                f"- Be in imperative mood (e.g., 'Add', 'Update', 'Fix', 'Remove')\n"
                f"- Be specific about what changed\n"
                f"- Be under 72 characters\n"
                f"- Focus on the most significant changes\n\n"
                f"File changes:\n{diff_output}\n\n"
                f"Detailed diff:\n{detailed_diff}\n\n"
                f"Generate only the commit message, nothing else:"
            )

            # Use the existing LLM service to generate the commit message
            response = await llm_service.generate_text(prompt, max_tokens=50)

            # Clean up the response
            commit_message = response.strip().strip('"').strip("'")

            # Fallback to a reasonable default if generation fails or is too long
            if not commit_message or len(commit_message) > 72:
                # Try to infer from file changes
                if "domain.yml" in diff_output:
                    return "Update domain configuration"
                elif any(f in diff_output for f in ["flows/", "data/flows/"]):
                    return "Update conversation flows"
                elif any(f in diff_output for f in ["nlu.yml", "data/nlu"]):
                    return "Update NLU training data"
                elif "config.yml" in diff_output:
                    return "Update model configuration"
                else:
                    return "Update bot files"

            return commit_message

        except Exception:
            structlogger.warning(
                "project_generator.commit_message_generation_failed",
                project_folder=self.project_folder.as_posix(),
            )
            # Fallback to generic message
            return "Update bot files"

    async def _get_current_branch(self) -> str:
        """Get the current Git branch name."""
        return await self.git_service.get_current_branch()

    async def checkout_branch(
        self, branch_name: str, create_if_not_exists: bool = False
    ) -> str:
        """Checkout a Git branch.

        Args:
            branch_name: Name of the branch to checkout
            create_if_not_exists: Whether to create the branch if it doesn't exist

        Raises:
            GitOperationInProgressError: If another git operation is in progress
            subprocess.CalledProcessError: If the checkout fails
        """
        # Ensure repository exists before branch operations
        if not self.git_service.git_dir.exists():
            self._ensure_git_repository()
        await self.git_service.checkout_branch(branch_name, create_if_not_exists)
        return await self.git_service.get_current_commit_sha()

    def bot_file_paths(
        self, exclude_models_directory: bool = True
    ) -> Generator[Path, None, None]:
        """Get the paths of all bot files."""
        for file in self.project_folder.glob("**/*"):
            # Skip directories
            if not file.is_file() or self.is_restricted_path(
                file, exclude_models_directory
            ):
                continue

            yield file

    def _get_bot_data_for_llm(self) -> Dict[str, Any]:
        """Get the current bot data for the LLM."""
        file_importer = self._create_importer()

        # only include data created by the user (or the builder llm)
        # avoid including to many defaults that are not customized
        domain = file_importer.get_user_domain()
        flows = file_importer.get_user_flows()

        return {
            "domain": domain.as_dict(should_clean_json=True),
            "flows": yaml_flows_io.get_flows_as_json(flows, should_clean_json=True),
        }

    def _path_for_flow(self, flow_id: str) -> str:
        """Get the path for a flow."""
        if flow_id.startswith("pattern_"):
            return f"data/patterns/{flow_id}.yml"
        else:
            return f"data/flows/{flow_id}.yml"

    async def _update_bot_files_from_llm_response(
        self, project_data: Dict[str, Any]
    ) -> None:
        """Update the bot files with generated data by writing to disk."""
        files = {"domain.yml": dump_obj_as_yaml_to_string(project_data["domain"])}
        # split up flows into one file per flow in the /flows folder
        for flow_id, flow_data in project_data["flows"].get("flows", {}).items():
            flow_file_path = self._path_for_flow(flow_id)
            single_flow_file_data = {"flows": {flow_id: flow_data}}
            files[flow_file_path] = dump_obj_as_yaml_to_string(single_flow_file_data)

        # removes any other flows that the LLM didn't generate
        self._cleanup_flows()
        commit_info = DEFAULT_COMMIT_INFO.model_copy(
            update={"message": "Update bot files"}
        )
        await self.update_bot_files(files, commit_info)

    def _cleanup_flows(self) -> None:
        """Cleanup the flows folder."""
        flows_folder = self.project_folder / "data" / "flows"
        if flows_folder.exists():
            shutil.rmtree(flows_folder)
        flows_folder.mkdir(parents=True, exist_ok=True)

    async def update_bot_files(
        self, files: Dict[str, Optional[str]], commit_info: GitCommitInfo
    ) -> None:
        """Update bot files with new content by writing to disk.

        Raises:
            GitOperationInProgressError: If another git operation is in progress
        """
        # Acquire lock for entire operation (file writes + commit)
        async with self.git_service.git_operation():
            for filename, content in files.items():
                file_path = Path(subpath(str(self.project_folder), filename))
                # Disallow updates inside .rasa project metadata directory
                if any(
                    part.startswith(".")
                    for part in file_path.relative_to(self.project_folder).parts
                ):
                    # silently ignore hidden paths
                    continue
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content or "", encoding="utf-8")

            # Commit changes to Git with AI-generated message
            # Note: _commit_changes uses the internal commit method which assumes
            # the lock is already held (which it is, by this context manager)
            await self._commit_changes(commit_info)

    def ensure_all_files_are_writable(self, files: Dict[str, Optional[str]]) -> None:
        """Ensure all files are writable."""
        for filename, content in files.items():
            file_path = Path(subpath(str(self.project_folder), filename))
            if self.is_restricted_path(file_path):
                raise InvalidPathException(
                    f"This file or folder is restricted from editing: {file_path}"
                )

    async def replace_all_bot_files(
        self, files: Dict[str, Optional[str]], commit_info: GitCommitInfo
    ) -> str:
        """Replace all bot files with new content, deleting files not in the request.

        Files/folders starting with .rasa/ or models/ are excluded from deletion.

        Args:
            files: Dictionary mapping file names to their content
            commit_info: info about the commit

        Returns:
            Commit SHA of the created commit

        Raises:
            GitOperationInProgressError: If another git operation is in progress
        """
        self.ensure_all_files_are_writable(files)

        # Acquire lock for entire operation (file writes/deletes + commit)
        async with self.git_service.git_operation():
            # Collect all existing files - any files not in the new `files` dict will be
            # deleted from this set
            existing_files = set(
                path.as_posix()
                for path in self.bot_file_paths(exclude_models_directory=True)
            )

            # Write all new files
            for filename, content in files.items():
                file_path = Path(subpath(str(self.project_folder), filename))
                file_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    file_path.write_text(content or "", encoding="utf-8")
                except Exception as e:
                    # Log write failure and avoid deleting an existing file by mistake
                    capture_exception_with_context(
                        e,
                        "project_generator.replace_all_bot_files.write_error",
                        extra={"file_path": file_path},
                    )
                    if file_path.as_posix() in existing_files:
                        # Keep the original file if it already existed
                        existing_files.discard(file_path.as_posix())
                    continue

                # Remove from deletion set since this file is in the new set of files
                existing_files.discard(file_path.as_posix())

            # Delete files that weren't in the request
            for file_to_delete in existing_files:
                file_path = Path(file_to_delete)
                try:
                    file_path.unlink()
                except Exception as e:
                    capture_exception_with_context(
                        e,
                        "project_generator.replace_all_bot_files.delete_error",
                        extra={"file_path": file_path},
                    )

            # Clean up empty directories (except excluded ones)
            self._cleanup_empty_directories()

            # Commit changes to Git with AI-generated message
            # Note: _commit_changes uses the internal commit method which assumes
            # the lock is already held (which it is, by this context manager)
            return await self._commit_changes(commit_info)

    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories from the project folder.

        Excludes hidden files and directories, and models/ from cleanup.
        """
        # Walk directories in reverse order (deepest first)
        for dirpath, dirnames, filenames in os.walk(self.project_folder, topdown=False):
            # Skip if this is the project root
            if dirpath == str(self.project_folder):
                continue

            if self.is_restricted_path(Path(dirpath)):
                continue

            relative_path = Path(dirpath).relative_to(self.project_folder)

            try:
                # Only remove if directory is empty
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
            except Exception as e:
                capture_exception_with_context(
                    e,
                    "project_generator.cleanup_empty_directories.error",
                    extra={"directory": relative_path.as_posix()},
                )

    def cleanup(self, skip_files: Optional[List[str]] = None) -> None:
        """Cleanup the project folder.

        Args:
            skip_files: List of file/directory names to skip during cleanup.
        """
        if skip_files is None:
            skip_files = []

        # Always include "lost+found" in skip files
        skip_files = list(skip_files) + ["lost+found"]

        # remove all the files and folders in the project folder resulting
        # in an empty folder
        for filename in os.listdir(self.project_folder):
            file_path = os.path.join(self.project_folder, filename)
            try:
                if filename in skip_files:
                    continue
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                structlogger.error(
                    "project_generator.cleanup_error",
                    error=str(e),
                    file_path=file_path,
                )
