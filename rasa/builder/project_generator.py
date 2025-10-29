"""Service for generating Rasa projects from prompts."""

import json
import os
import shutil
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Generator, List, Optional

import structlog

from rasa.builder import config
from rasa.builder.exceptions import ProjectGenerationError, ValidationError
from rasa.builder.llm_service import get_skill_generation_messages, llm_service
from rasa.builder.logging_utils import capture_exception_with_context
from rasa.builder.models import BotFiles
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

    async def init_from_template(self, template: ProjectTemplateName) -> None:
        """Create the initial project files."""
        self.cleanup()
        create_initial_project(self.project_folder.as_posix(), template)
        # If a local cache for this template exists, copy it into the project.
        # We no longer download here to avoid blocking project creation.
        await copy_cache_for_template_if_available(template, self.project_folder)
        # needs to happen after caching, as we download/copy .rasa and that would
        # overwrite the project info file in .rasa
        ensure_first_used(self.project_folder)

    async def generate_project_with_retries(
        self,
        skill_description: str,
        template: ProjectTemplateName,
        max_retries: Optional[int] = None,
    ) -> Dict[str, Optional[str]]:
        """Generate a Rasa project with retry logic for validation failures.

        Args:
            skill_description: Natural language description of the skill
            rasa_config: Rasa configuration dictionary
            template: Project template to use for the initial project
            max_retries: Maximum number of retry attempts

        Returns:
            Dictionary of generated file contents (filename -> content)

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
        ) -> Dict[str, Optional[str]]:
            try:
                # Generate project data using LLM
                project_data = await llm_service.generate_rasa_project(messages)

                # Update stored bot data
                self._update_bot_files_from_llm_response(project_data)

                bot_files = self.get_bot_files()
                structlogger.info(
                    "project_generator.generated_project",
                    attempts_left=attempts_left,
                    files=list(bot_files.keys()),
                )

                # Validate the generated project
                await self._validate_generated_project()

                structlogger.info(
                    "project_generator.validation_success", attempts_left=attempts_left
                )

                return bot_files

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

                return await _generate_with_retry(
                    error_feedback_messages, attempts_left - 1
                )

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
                return await _generate_with_retry(initial_messages, attempts_left - 1)

        return await _generate_with_retry(initial_messages, max_retries)

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

    def _update_bot_files_from_llm_response(self, project_data: Dict[str, Any]) -> None:
        """Update the bot files with generated data by writing to disk."""
        files = {"domain.yml": dump_obj_as_yaml_to_string(project_data["domain"])}
        # split up flows into one file per flow in the /flows folder
        for flow_id, flow_data in project_data["flows"].get("flows", {}).items():
            flow_file_path = self._path_for_flow(flow_id)
            single_flow_file_data = {"flows": {flow_id: flow_data}}
            files[flow_file_path] = dump_obj_as_yaml_to_string(single_flow_file_data)

        # removes any other flows that the LLM didn't generate
        self._cleanup_flows()
        self.update_bot_files(files)

    def _cleanup_flows(self) -> None:
        """Cleanup the flows folder."""
        flows_folder = self.project_folder / "data" / "flows"
        if flows_folder.exists():
            shutil.rmtree(flows_folder)
        flows_folder.mkdir(parents=True, exist_ok=True)

    def update_bot_files(self, files: Dict[str, Optional[str]]) -> None:
        """Update bot files with new content by writing to disk."""
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
            file_path.write_text(content, encoding="utf-8")

    def ensure_all_files_are_writable(self, files: Dict[str, Optional[str]]) -> None:
        """Ensure all files are writable."""
        for filename, content in files.items():
            file_path = Path(subpath(str(self.project_folder), filename))
            if self.is_restricted_path(file_path):
                raise InvalidPathException(
                    f"This file or folder is restricted from editing: {file_path}"
                )

    def replace_all_bot_files(self, files: Dict[str, Optional[str]]) -> None:
        """Replace all bot files with new content, deleting files not in the request.

        Files/folders starting with .rasa/ or models/ are excluded from deletion.

        Args:
            files: Dictionary mapping file names to their content
        """
        self.ensure_all_files_are_writable(files)
        # Collect all existing files - any files not in the new `files` dict will be
        # deleted from this set
        existing_files = set(
            path.as_posix()
            for path in self.bot_file_paths(exclude_models_directory=True)
        )

        # Write all new files
        for filename, content in files.items():
            if content is None:
                continue

            file_path = Path(subpath(str(self.project_folder), filename))
            file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                file_path.write_text(content, encoding="utf-8")
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
