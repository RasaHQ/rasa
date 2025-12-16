"""Service for generating Rasa projects from prompts."""

import asyncio
import json
import os
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple, cast

import importlib_resources
import openai
import structlog
from jinja2 import Template
from openai.types.chat import ChatCompletion

from rasa.builder import config
from rasa.builder.config import PROJECT_GENERATION_TIMEOUT
from rasa.builder.copilot import Copilot
from rasa.builder.copilot.constants import (
    DEFAULT_COMMIT_MESSAGE,
)
from rasa.builder.copilot.models import (
    CopilotContext,
    FileContent,
    InternalCopilotRequestChatMessage,
    LogContent,
    ResponseCategory,
    UsageStatistics,
)
from rasa.builder.exceptions import (
    LLMGenerationError,
    ProjectGenerationError,
    ValidationError,
)
from rasa.builder.git_service import DEFAULT_COMMIT_INFO, GitService
from rasa.builder.logging_utils import capture_exception_with_context
from rasa.builder.models import BotFiles, GitCommitInfo
from rasa.builder.project_generator.project_utils import (
    bot_file_paths,
    get_bot_files,
    is_restricted_path,
    path_relative_to_project,
    unsafe_write_to_bot_files,
)
from rasa.builder.project_info import ProjectInfo, ensure_first_used, load_project_info
from rasa.builder.telemetry.commit_langfuse_telemetry import (
    CommitMessageGenerationLangfuseTelemetry,
)
from rasa.builder.telemetry.langfuse_compat import observe
from rasa.builder.telemetry.prompt_to_bot_langfuse_telemetry import (
    PromptToBotLangfuseTelemetry,
)
from rasa.builder.telemetry.welcome_langfuse_telemetry import (
    WelcomeMessageGenerationLangfuseTelemetry,
)
from rasa.builder.template_cache import copy_cache_for_template_if_available
from rasa.builder.training_service import TrainingInput
from rasa.builder.validation_service import validate_project
from rasa.cli.scaffold import ProjectTemplateName, create_initial_project
from rasa.shared.constants import (
    DOMAIN_SCHEMA_FILE,
    PACKAGE_NAME,
    RESPONSES_SCHEMA_FILE,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_USER,
)
from rasa.shared.core.flows import yaml_flows_io
from rasa.shared.core.flows.yaml_flows_io import FLOWS_SCHEMA_FILE
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.utils.io import read_json_file
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string, read_schema_file
from rasa.utils.io import InvalidPathException

structlogger = structlog.get_logger()

BULLET_POINT_REGEX = re.compile(r"^-\s+\*[^*]+\*\s*$")


class ProjectGenerator:
    """Service for generating Rasa projects from skill descriptions."""

    def __init__(self, project_folder: str) -> None:
        """Initialize the project generator with a folder for file persistence.

        Args:
            project_folder: Path to the folder where project files will be stored
        """
        self._client: Optional[openai.AsyncOpenAI] = None
        self.project_folder = Path(project_folder)
        self.project_folder.mkdir(parents=True, exist_ok=True)
        self.git_service = GitService(project_folder)
        self._skill_to_bot_system_prompt = Template(
            importlib_resources.read_text(  # type: ignore[no-untyped-call]
                "rasa.builder.project_generator.prompts",
                "skill_to_bot_system_prompt.jinja2",
            )
        )
        self._skill_to_bot_user_prompt = Template(
            importlib_resources.read_text(  # type: ignore[no-untyped-call]
                "rasa.builder.project_generator.prompts",
                "skill_to_bot_user_request_prompt.jinja2",
            )
        )
        self._error_feedback_template = Template(
            importlib_resources.read_text(  # type: ignore[no-untyped-call]
                "rasa.builder.project_generator.prompts",
                "skill_to_bot_error_feedback_prompt.jinja2",
            )
        )
        self._welcome_message_prompt_template = Template(
            importlib_resources.read_text(  # type: ignore[no-untyped-call]
                "rasa.builder.copilot.prompts",
                "welcome_message_prompt.jinja2",
            )
        )
        self._commit_message_prompt_template = Template(
            importlib_resources.read_text(  # type: ignore[no-untyped-call]
                "rasa.builder.copilot.prompts",
                "commit_message_prompt.jinja2",
            )
        )

        # Initialize the lists for storing the generated project data and usage
        # statistics. These are used to create error feedback messages for the next
        # attempt. Kind of like a conversation history.
        self._generated_project_data_attempts: List[Dict[str, Any]] = []
        self._usage_statistics_attempts: List[UsageStatistics] = []

        # Get the domain and flows schemas for structuring the LLM response with
        # the structured output.
        self._domain_schema = self._get_domain_schema()
        self._flows_schema = self._get_flows_schema()

        # Retrieved documentation from InKeep AI. We make it static to make the most
        # use of the input tokens caching.
        self._flow_documentation = self._get_flow_documentation()
        self._domain_documentation = self._get_domain_documentation()
        self._custom_actions_documentation = self._get_custom_actions_documentation()

        # Migrate existing projects to Git if needed
        self.migrate_git_repository_if_needed()

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Get or lazy create OpenAI client with proper resource management."""
        if self._client is None:
            self._client = openai.AsyncOpenAI()

        try:
            yield self._client
        except Exception as e:
            structlogger.error("project_generator.llm_client_error", error=str(e))
            raise

    def _get_domain_schema(self) -> Dict[str, Any]:
        """Return a modified domain schema dictionary for project generation."""
        domain_schema = deepcopy(
            read_schema_file(DOMAIN_SCHEMA_FILE, PACKAGE_NAME, False)
        )

        if not isinstance(domain_schema, dict):
            raise ValueError("Domain schema is not a dictionary")

        # Remove parts not needed for CALM bots
        unnecessary_keys = ["intents", "entities", "forms", "config", "session_config"]
        mapping_obj = domain_schema.get("mapping")
        if not isinstance(mapping_obj, dict):
            raise ValueError("Domain schema mapping is not a dictionary")
        mapping = cast(Dict[str, Any], mapping_obj)
        for key in unnecessary_keys:
            mapping.pop(key, None)

        # Remove problematic slot mappings
        slots_obj = mapping.get("slots")
        if isinstance(slots_obj, dict):
            slots_dict = cast(Dict[str, Any], slots_obj)
            slots_mapping_obj = cast(
                Optional[Dict[str, Any]], slots_dict.get("mapping")
            )
            if slots_mapping_obj:
                regex_slot_obj = cast(
                    Optional[Dict[str, Any]],
                    slots_mapping_obj.get("regex;([A-Za-z]+)"),
                )
                if regex_slot_obj:
                    slot_mapping_obj = cast(
                        Optional[Dict[str, Any]], regex_slot_obj.get("mapping")
                    )
                    if slot_mapping_obj:
                        slot_mapping_obj.pop("mappings", None)
                        slot_mapping_obj.pop("validation", None)

        # Add responses schema
        responses_schema_data = read_schema_file(
            RESPONSES_SCHEMA_FILE, PACKAGE_NAME, False
        )
        if not isinstance(responses_schema_data, dict):
            raise ValueError("Responses schema file is not a dictionary")
        responses_schema_raw = cast(
            Optional[Dict[str, Any]],
            responses_schema_data.get("schema;responses"),
        )
        if responses_schema_raw is None:
            raise ValueError("Responses schema is not a dictionary")
        mapping["responses"] = responses_schema_raw

        return domain_schema

    def _get_flows_schema(self) -> Dict[str, Any]:
        """Return a modified flows schema dictionary for project generation."""
        schema_file = str(
            importlib_resources.files(PACKAGE_NAME).joinpath(FLOWS_SCHEMA_FILE)
        )
        flows_schema = deepcopy(read_json_file(schema_file))
        flows_schema["$defs"]["flow"]["properties"].pop("nlu_trigger", None)
        return flows_schema

    def _get_flow_documentation(self) -> str:
        """Get the flow documentation."""
        return importlib_resources.read_text(  # type: ignore[no-untyped-call]
            "rasa.builder.project_generator.prompts",
            "flow_documentation.json",
        )

    def _get_domain_documentation(self) -> str:
        """Get the domain documentation."""
        return importlib_resources.read_text(  # type: ignore[no-untyped-call]
            "rasa.builder.project_generator.prompts",
            "domain_documentation.json",
        )

    def _get_custom_actions_documentation(self) -> str:
        """Get the custom actions documentation."""
        return importlib_resources.read_text(  # type: ignore[no-untyped-call]
            "rasa.builder.project_generator.prompts",
            "custom_actions_documentation.json",
        )

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
        """Create the initial project files."""
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
        return await self.git_service.commit_changes(
            DEFAULT_COMMIT_INFO.model_copy(
                update={"message": f"Initialize project from {template.value} template"}
            )
        )

    async def generate_project_with_retries(
        self,
        skill_description: str,
        template: ProjectTemplateName,
        max_retries: Optional[int] = None,
    ) -> Tuple[int, str]:
        """Generate a Rasa project with retry logic for validation failures.

        Args:
            skill_description: Natural language description of the skill
            template: Project template to use for the initial project
            max_retries: Maximum number of retry attempts

        Returns:
            A tuple containing the
            - number of attempts taken
            - the commit SHA of the generated project

        Raises:
            ProjectGenerationError: If generation fails after all retries
        """
        if max_retries is None:
            max_retries = config.PROJECT_GENERATION_MAX_RETRIES

        # if ever we do not use the template, we need to ensure first_used is set
        # separately!
        commit_sha = await self.init_from_template(template)
        self._generated_project_data_attempts = []
        self._usage_statistics_attempts = []
        project_data = self._get_bot_data_for_llm()
        system_message = self._create_system_message()
        user_request_message = self._create_user_request_message(
            skill_description, project_data
        )
        initial_messages = [system_message, user_request_message]
        error_feedback_messages: List[Dict[str, Any]] = []

        attempts_left = max_retries
        while attempts_left > 0:
            try:
                commit_sha = await self._attempt_generation(
                    initial_messages=initial_messages,
                    error_feedback_messages=error_feedback_messages,
                    attempts_left=attempts_left,
                    max_retries=max_retries,
                )
                return max_retries - attempts_left, commit_sha

            # The generation attempt failed due to validation errors
            except ValidationError as e:
                if (attempts_left := attempts_left - 1) <= 0:
                    return max_retries, commit_sha
                    raise ProjectGenerationError(
                        f"Failed to generate valid Rasa project: {e}", max_retries
                    ) from e

                # Use the last generated project data, or fall back to original
                # if list is empty
                project_data_for_feedback = (
                    self._generated_project_data_attempts[-1]
                    if self._generated_project_data_attempts
                    else project_data
                )

                # Get copilot guidance for the error
                copilot_guidance = await self._get_copilot_error_guidance(
                    error=e, project_data=project_data_for_feedback
                )

                # Create error feedback message
                error_feedback_message = self._create_error_feedback_messages(
                    project_data=project_data_for_feedback,
                    error=e,
                    copilot_guidance=copilot_guidance,
                )
                error_feedback_messages.extend(error_feedback_message)

            # The generation attempt failed due to other errors
            except Exception as e:
                if (attempts_left := attempts_left - 1) <= 0:
                    raise ProjectGenerationError(
                        f"Failed to generate Rasa project: {e}", max_retries
                    ) from e
                # Reset error feedback messages for non-validation errors
                error_feedback_messages = []

        # This should never be reached, but satisfies type checker
        raise ProjectGenerationError(
            "Failed to generate Rasa project: exhausted all retry attempts", max_retries
        )

    @observe()
    async def _attempt_generation(
        self,
        initial_messages: List[Dict[str, Any]],
        error_feedback_messages: List[Dict[str, Any]],
        attempts_left: int,
        max_retries: int,
    ) -> str:
        """Attempt to generate and validate a project in a single try."""
        generated_project_data: Optional[Dict[str, Any]] = None
        bot_files: Optional[Dict[str, Optional[str]]] = None

        try:
            # Generate the project data
            generated_project_data = await self.generate_response(
                initial_messages=initial_messages,
                error_feedback_messages=error_feedback_messages,
            )
            # Append the generated project data to the list of attempts. This is used
            # create error feedback messages for the next attempt
            self._generated_project_data_attempts.append(generated_project_data)
            # Update the bot files from the generated project data
            commit_sha = await self._update_bot_files_from_llm_response(
                generated_project_data
            )

            bot_files = self.get_bot_files()
            structlogger.info(
                "project_generator.generated_project",
                attempts_left=attempts_left,
                max_retries=max_retries,
                files=list(bot_files.keys()),
            )

            await self._validate_generated_project()
            structlogger.info(
                "project_generator.validation_success",
                attempts_left=attempts_left,
                max_retries=max_retries,
            )

            return commit_sha

        except ValidationError as e:
            structlogger.error(
                "project_generator.validation_error",
                error=str(e),
                attempts_left=attempts_left,
                max_retries=max_retries,
            )
            raise e

        except Exception as e:
            structlogger.error(
                "project_generator.generation_error",
                error=str(e),
                attempts_left=attempts_left,
                max_retries=max_retries,
            )
            raise e

    @observe(as_type="generation")
    async def generate_response(
        self,
        initial_messages: List[Dict[str, Any]],
        error_feedback_messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Generate Rasa assistant project data via the LLM."""
        try:
            async with self._get_client() as client:
                messages = [*initial_messages, *error_feedback_messages]
                response_format: Dict[str, Any] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "rasa_project",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "domain": self._domain_schema,
                                "flows": self._flows_schema,
                            },
                            "required": ["domain", "flows"],
                        },
                    },
                }
                response = await client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=cast(Any, messages),
                    timeout=PROJECT_GENERATION_TIMEOUT,
                    temperature=config.OPENAI_TEMPERATURE,
                    response_format=cast(Any, response_format),
                )

            # Extract and store usage statistics
            usage_statistics = UsageStatistics.from_chat_completion_response(
                response,
                input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
            )
            if usage_statistics:
                self._usage_statistics_attempts.append(usage_statistics)
                # Update the current Langfuse span with usage statistics
                PromptToBotLangfuseTelemetry.update_current_span_with_usage_statistics(
                    usage_statistics
                )

            content = response.choices[0].message.content
            if not content:
                error_message = "Empty response from LLM."
                structlogger.error(
                    "project_generator.generate_response.empty_llm_response",
                    event_info=error_message,
                    messages=messages,
                    response=response,
                )
                raise LLMGenerationError(error_message)

            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                error_message = f"Invalid JSON from LLM: {e}."
                structlogger.error(
                    "project_generator.generate_response.llm_response_invalid_json_format",
                    content=content,
                    event_info=error_message,
                    error=str(e),
                )
                raise LLMGenerationError(error_message)

        except openai.OpenAIError as e:
            error_message = f"OpenAI API error: {e}."
            structlogger.error(
                "project_generator.generate_response.openai_api_error",
                event_info=error_message,
                messages=messages,
                error=str(e),
            )
            raise LLMGenerationError(error_message)
        except asyncio.TimeoutError as e:
            error_message = "LLM request timed out."
            structlogger.error(
                "project_generator.generate_response.llm_request_timeout",
                event_info=error_message,
                messages=messages,
                error=str(e),
            )
            raise LLMGenerationError(error_message)

    @observe()
    async def _get_copilot_error_guidance(
        self,
        error: Exception,
        project_data: Dict[str, Any],
    ) -> Optional[str]:
        """Get copilot error analysis guidance for validation errors.

        Args:
            error: The validation error that occurred
            project_data: The project data that failed validation

        Returns:
            Copilot guidance text if available, None otherwise
        """
        try:
            # Build error message from exception
            logs = error.validation_logs if isinstance(error, ValidationError) else []
            error_message = str(error)
            if logs:
                error_message += "\n\n" + "\n".join(str(log) for log in logs)

            # Convert project_data to file content blocks for copilot analysis
            file_content_blocks: List[FileContent] = []
            # Add domain.yml
            domain_content = dump_obj_as_yaml_to_string(project_data["domain"])
            file_content_blocks.append(
                FileContent(
                    type="file", file_path="domain.yml", file_content=domain_content
                )
            )
            # Add flow files
            flows_dict = project_data.get("flows", {})
            for flow_id, flow_data in flows_dict.items():
                flow_file_path = self._path_for_flow(flow_id)
                single_flow_file_data = {"flows": {flow_id: flow_data}}
                flow_content = dump_obj_as_yaml_to_string(single_flow_file_data)
                file_content_blocks.append(
                    FileContent(
                        type="file", file_path=flow_file_path, file_content=flow_content
                    )
                )

            # Create message content blocks with log content and available files
            log_content_block = LogContent(
                type="log", content=error_message, context="validation_error"
            )
            context = CopilotContext(
                tracker_context=None,  # No conversation context needed
                copilot_chat_history=[
                    InternalCopilotRequestChatMessage(
                        role="internal_copilot_request",
                        content=[log_content_block, *file_content_blocks],
                        response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
                    )
                ],
            )

            copilot = Copilot()
            # Generate copilot response and handle it with the response handler.
            # Consume the stream to get the full response.
            (
                copilot_response_handler,
                generation_context,
            ) = await copilot.generate_response(context)
            async for _ in copilot_response_handler.stream():
                pass

            # Extract the full text from the handler
            full_text = copilot_response_handler.extract_full_text()
            return full_text if full_text else None

        except Exception as e:
            structlogger.warning(
                "project_generator.copilot_error_analysis_failed",
                error=str(e),
                event_info=(
                    "Failed to get copilot error guidance, continuing without it"
                ),
            )
            return None

    def _create_system_message(self) -> Dict[str, Any]:
        """Create a system message for skill generation."""
        system_prompt = self._skill_to_bot_system_prompt.render(
            flow_documentation_results=self._flow_documentation,
            domain_documentation_results=self._domain_documentation,
            custom_actions_documentation_results=self._custom_actions_documentation,
        )
        return {"role": ROLE_SYSTEM, "content": system_prompt}

    def _create_user_request_message(
        self, skill_description: str, project_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a user request message for skill generation."""
        user_request_prompt = self._skill_to_bot_user_prompt.render(
            skill_description=skill_description,
            project_data=project_data,
        )
        return {"role": ROLE_USER, "content": user_request_prompt}

    def _create_error_feedback_messages(
        self,
        project_data: Dict[str, Any],
        error: Exception,
        copilot_guidance: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Create error feedback messages for the LLM.

        Args:
            project_data: The project data that failed validation
            error: The validation error that occurred
            copilot_guidance: Optional copilot error analysis guidance
        """
        logs = error.validation_logs if isinstance(error, ValidationError) else []
        error_feedback_prompt = self._error_feedback_template.render(
            project_data=project_data,
            error=error,
            logs=logs,
            copilot_guidance=copilot_guidance,
        )
        return [
            {
                "role": ROLE_ASSISTANT,
                "content": json.dumps(project_data),
            },
            {
                "role": ROLE_USER,
                "content": error_feedback_prompt,
            },
        ]

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
        """Get the current bot files by reading from disk."""
        return get_bot_files(
            self.project_folder,
            allowed_file_extensions,
            exclude_docs_directory,
            exclude_models_directory,
        )

    def bot_file_paths(
        self, exclude_models_directory: bool = True
    ) -> Generator[Path, None, None]:
        """Get the paths of all bot files."""
        yield from bot_file_paths(self.project_folder, exclude_models_directory)

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
    ) -> str:
        """Update the bot files with generated data by writing to disk."""
        files: Dict[str, Optional[str]] = {
            "domain/domain.yml": dump_obj_as_yaml_to_string(project_data["domain"])
        }
        # split up flows into one file per flow in the /flows folder
        for flow_id, flow_data in project_data.get("flows", {}).items():
            flow_file_path = self._path_for_flow(flow_id)
            single_flow_file_data = {"flows": {flow_id: flow_data}}
            files[flow_file_path] = dump_obj_as_yaml_to_string(single_flow_file_data)

        # removes any other flows that the LLM didn't generate
        self._cleanup_flows()
        commit_sha = await self.update_bot_files(files)
        return commit_sha

    def _cleanup_flows(self) -> None:
        """Cleanup the flows folder."""
        flows_folder = self.project_folder / "data" / "flows"
        if flows_folder.exists():
            shutil.rmtree(flows_folder)
        flows_folder.mkdir(parents=True, exist_ok=True)

    async def update_bot_files(self, files: Dict[str, Optional[str]]) -> str:
        """Update bot files with new content by writing to disk."""
        # Acquire lock for entire operation (file writes + commit)
        async with self.git_service.git_operation():
            # Ensure git repository exists before committing
            if not self.git_service.git_dir.exists():
                self._ensure_git_repository()

            unsafe_write_to_bot_files(
                self.project_folder, files, fail_on_restricted_path=False
            )
            # Commit changes using internal method (lock already held)
            commit_sha = await self.git_service._commit_changes_internal(
                DEFAULT_COMMIT_INFO.model_copy(
                    update={"message": "Generated initial project files"}
                )
            )
            return commit_sha

    def ensure_all_files_are_writable(self, files: Dict[str, Optional[str]]) -> None:
        """Ensure all files are writable."""
        for filename, content in files.items():
            file_path = path_relative_to_project(self.project_folder, filename)
            if is_restricted_path(self.project_folder, file_path):
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
            try:
                file_path = path_relative_to_project(self.project_folder, filename)
                unsafe_write_to_bot_files(
                    self.project_folder,
                    {filename: content},
                    fail_on_restricted_path=False,
                )
                # Remove from deletion set since this file is
                # in the new set of files
                existing_files.discard(file_path.as_posix())
            except Exception as e:
                # Log write failure and avoid deleting an existing file by mistake
                capture_exception_with_context(
                    e,
                    "project_generator.replace_all_bot_files.write_error",
                    extra={"file_path": filename},
                )
                continue

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
        return await self.unsafe_commit_changes(commit_info)

    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories from the project folder.

        Excludes hidden files and directories, and models/ from cleanup.
        """
        # Walk directories in reverse order (deepest first)
        for dirpath, dirnames, filenames in os.walk(self.project_folder, topdown=False):
            # Skip if this is the project root
            if dirpath == str(self.project_folder):
                continue

            if is_restricted_path(self.project_folder, Path(dirpath)):
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

    async def unsafe_commit_changes(self, commit_info: GitCommitInfo) -> str:
        """Commit all changes.

        Args:
            commit_info: info about the commit

        Returns:
            Commit SHA of the created commit
        """
        try:
            # Ensure git repository exists (defensive in case migration didn't run)
            if not self.git_service.git_dir.exists():
                self._ensure_git_repository()
            # Generate commit message if not provided
            if commit_info.message is None:
                commit_info.message = await self._generate_commit_message()

            # Commit changes using GitService
            commit_sha = await self.git_service.commit_changes(commit_info)

            return commit_sha

        except Exception as e:
            structlogger.error(
                "project_generator.git_commit_failed",
                error=str(e),
                project_folder=self.project_folder.as_posix(),
            )
            # Don't fail the operation if Git commit fails, return current commit
            return await self.git_service.get_current_commit_sha()

    @WelcomeMessageGenerationLangfuseTelemetry.trace_text_generation
    async def _generate_text(
        self, prompt: str, max_completion_tokens: int = 100
    ) -> ChatCompletion:
        """Generate simple text using OpenAI.

        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum tokens to generate

        Returns:
            Chat Completion response

        Raises:
            LLMGenerationError: If generation fails
        """
        try:
            async with self._get_client() as client:
                response = await client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Lower temperature for consistent messages
                    max_completion_tokens=max_completion_tokens,
                )

                if not response.choices[0].message.content:
                    raise LLMGenerationError("Empty response from LLM")

                return response

        except openai.OpenAIError as e:
            raise LLMGenerationError(f"OpenAI API error: {e}")
        except asyncio.TimeoutError:
            raise LLMGenerationError("LLM request timed out")

    @observe()
    async def _generate_commit_message(self) -> str:
        """Generate a meaningful commit message using AI based on the changes.

        Returns:
            A descriptive commit message based on the Git diff
        """
        try:
            # Get the diff of staged changes
            diff_output = (
                await self.git_service.run_git_command(
                    ["diff", "--name-status"], check_output=True
                )
                or ""
            )
            if not diff_output:
                return DEFAULT_COMMIT_MESSAGE

            # Get a more detailed diff for context (limited to avoid token limits)
            detailed_diff = (
                await self.git_service.run_git_command(
                    ["diff", "--unified=2"], check_output=True
                )
                or ""
            )

            # Limit the diff size to avoid token limits
            if detailed_diff and len(detailed_diff) > 2000:
                detailed_diff = detailed_diff[:2000] + "\n... (diff truncated)"

            # Render the prompt using the template
            prompt = self._commit_message_prompt_template.render(
                diff_output=diff_output,
                detailed_diff=detailed_diff,
                desired_length=config.COMMIT_MESSAGE_DESIRED_LENGTH,
            )

            # Update Langfuse span with input data
            CommitMessageGenerationLangfuseTelemetry.update_commit_message_generation_input(
                diff_output=diff_output,
                detailed_diff=detailed_diff,
                prompt=prompt,
            )

            # Use the existing LLM service to generate the commit message
            response = await self._generate_text(
                prompt, max_completion_tokens=config.COMMIT_MESSAGE_MAX_TOKENS
            )
            response_content = response.choices[0].message.content or ""

            # Clean up the response
            commit_message = response_content.strip().strip('"').strip("'")

            # Fallback to a reasonable default if generation fails or is too long
            if (
                not commit_message
                or len(commit_message) > config.COMMIT_MESSAGE_MAX_CHARACTERS
            ):
                commit_message = DEFAULT_COMMIT_MESSAGE

            # Update Langfuse span with output data
            CommitMessageGenerationLangfuseTelemetry.update_commit_message_generation_output(
                response_content=response_content,
                commit_message=commit_message,
            )

            return commit_message

        except Exception:
            structlogger.warning(
                "project_generator.commit_message_generation_failed",
                project_folder=self.project_folder.as_posix(),
            )
            # Fallback to generic message
            return DEFAULT_COMMIT_MESSAGE

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

    async def _get_current_branch(self) -> str:
        """Get the current Git branch name."""
        return await self.git_service.get_current_branch()

    def _verify_bullet_points(self, response: str, max_amount: int) -> bool:
        """Verify that the response is in bullet point format."""
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        if not lines:
            return False
        if len(lines) > max_amount:
            return False
        return all(BULLET_POINT_REGEX.match(line) for line in lines)

    @observe()
    async def generate_welcome_message(
        self, default_welcome_message: str, template_welcome_message: str
    ) -> str:
        """Generate a welcome message based on the generated flows.

        Returns:
            welcome message string
        """
        try:
            # Get generated flows
            flows = self._get_bot_data_for_llm().get("flows", {})
            if not flows:
                return default_welcome_message

            # Render the prompt using the template
            prompt = self._welcome_message_prompt_template.render(flows=flows)

            # Update Langfuse span with input data
            WelcomeMessageGenerationLangfuseTelemetry.update_welcome_message_generation_input(
                flows=flows,
                prompt=prompt,
            )

            # Use the existing LLM service to generate the example questions
            response = await self._generate_text(
                prompt, max_completion_tokens=config.WELCOME_MESSAGE_MAX_TOKENS
            )
            response_content = response.choices[0].message.content or ""

            # Initialize variables with defaults
            welcome_message = default_welcome_message

            # Clean up the response
            if (
                response_content
                and template_welcome_message
                and self._verify_bullet_points(response_content, 3)
            ):
                welcome_message = template_welcome_message.format(
                    example_questions=response_content
                )

            # Update Langfuse span with output data
            WelcomeMessageGenerationLangfuseTelemetry.update_welcome_message_generation_output(
                response_content=response_content,
                welcome_message=welcome_message,
            )

            return welcome_message

        except Exception:
            structlogger.warning(
                "project_generator.welcome_message_generation_failed",
                project_folder=self.project_folder.as_posix(),
            )
            # Fallback to generic message
            return default_welcome_message
