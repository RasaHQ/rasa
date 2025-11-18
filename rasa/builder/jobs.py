import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from sanic import Sanic

from rasa.builder import config
from rasa.builder.constants import (
    MAX_ARCHIVE_FILE_SIZE,
    MAX_ARCHIVE_FILES,
    MAX_ARCHIVE_TOTAL_SIZE,
)
from rasa.builder.copilot.constants import (
    PROMPT_TO_BOT_KEY,
)
from rasa.builder.copilot.copilot_templated_message_provider import (
    load_copilot_handler_default_responses,
    load_copilot_template_prompts,
    load_copilot_welcome_messages,
)
from rasa.builder.copilot.history_store import (
    persist_copilot_message_to_history,
    persist_training_error_analysis_to_history,
    persist_user_message_to_history,
)
from rasa.builder.copilot.models import (
    CopilotContext,
    FileContent,
    InternalCopilotRequestChatMessage,
    LogContent,
    ResponseCategory,
    TrainingErrorLog,
)
from rasa.builder.download import download_backup_from_url
from rasa.builder.exceptions import (
    LLMGenerationError,
    ProjectGenerationError,
    TrainingError,
    ValidationError,
)
from rasa.builder.job_manager import JobInfo, job_manager
from rasa.builder.llm_service import llm_service
from rasa.builder.models import (
    JobStatus,
    JobStatusEvent,
)
from rasa.builder.project_generator import ProjectGenerator
from rasa.builder.training_service import (
    train_and_load_agent,
    try_load_existing_agent,
    update_agent,
)
from rasa.builder.validation_service import validate_project
from rasa.cli.scaffold import ProjectTemplateName
from rasa.core.agent import load_agent
from rasa.core.config.configuration import Configuration
from rasa.exceptions import ModelNotFound
from rasa.model import get_local_model
from rasa.shared.constants import DEFAULT_ENDPOINTS_PATH

structlogger = structlog.get_logger()


async def push_job_status_event(
    job: JobInfo,
    status: JobStatus,
    message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    event = JobStatusEvent.from_status(
        status=status.value, message=message, payload=payload
    )
    job.status = status.value
    await job.put(event)


async def run_prompt_to_bot_job(
    app: Any,
    job: JobInfo,
    prompt: str,
) -> None:
    """Run the prompt-to-bot job in the background.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        prompt: The natural language prompt for bot generation.
    """
    project_generator: ProjectGenerator = app.ctx.project_generator

    await push_job_status_event(job, JobStatus.received)

    try:
        # 1. Generating
        await push_job_status_event(job, JobStatus.generating)
        bot_files = await project_generator.generate_project_with_retries(
            prompt,
            template=ProjectTemplateName.BASIC,
        )
        await push_job_status_event(job, JobStatus.generation_success)

        # 2. Training
        await push_job_status_event(job, JobStatus.training)
        agent = await train_and_load_agent(project_generator.get_training_input())
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        # 3. Create copilot welcome message job
        copilot_welcome_job = job_manager.create_job()
        app.add_task(run_copilot_welcome_message_job(app, copilot_welcome_job))

        structlogger.info(
            "bot_builder_service.prompt_to_bot.success",
            files_generated=list(bot_files.keys()),
            copilot_welcome_job_id=copilot_welcome_job.id,
        )
        await push_job_status_event(
            job=job,
            status=JobStatus.done,
            payload={"copilot_welcome_job_id": copilot_welcome_job.id},
        )
        job_manager.mark_done(job)

    except TrainingError as exc:
        structlogger.debug(
            "prompt_to_bot_job.training_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.train_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except ValidationError as exc:
        # Log levels to include in the error message
        log_levels = ["error"]
        if config.VALIDATION_FAIL_ON_WARNINGS:
            log_levels.append("warning")

        structlogger.debug(
            "prompt_to_bot_job.validation_error",
            job_id=job.id,
            error=str(exc),
            all_validation_logs=exc.validation_logs,
            included_log_levels=log_levels,
        )

        error_message = exc.get_error_message_with_logs(log_levels=log_levels)
        await push_job_status_event(
            job, JobStatus.validation_error, message=error_message
        )
        job_manager.mark_done(job, error=error_message)

    except (ProjectGenerationError, LLMGenerationError) as exc:
        structlogger.debug(
            "prompt_to_bot_job.generation_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.generation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except Exception as exc:
        # Capture full traceback
        structlogger.exception(
            "prompt_to_bot_job.unexpected_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


async def run_template_to_bot_job(
    app: "Sanic",
    job: JobInfo,
    template_name: ProjectTemplateName,
) -> None:
    """Run the template-to-bot job in the background.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        template_name: The name of the template to use for bot generation.
    """
    project_generator: ProjectGenerator = app.ctx.project_generator

    copilot_template_prompt_job = job_manager.create_job()
    await push_job_status_event(
        job,
        JobStatus.received,
        payload={"copilot_template_prompt_job_id": copilot_template_prompt_job.id},
    )

    try:
        app.add_task(
            run_copilot_template_prompt_job(
                app, copilot_template_prompt_job, template_name
            )
        )

        await push_job_status_event(job, JobStatus.generating)
        await project_generator.init_from_template(template_name)
        bot_files = project_generator.get_bot_files()
        await push_job_status_event(job, JobStatus.generation_success)

        await push_job_status_event(job, JobStatus.training)
        agent = await try_load_existing_agent(project_generator.project_folder)
        if agent is None:
            agent = await train_and_load_agent(project_generator.get_training_input())
        else:
            structlogger.info(
                "bot_builder_service.template_to_bot.agent_loaded_from_cache",
            )
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        copilot_welcome_job = job_manager.create_job()
        app.add_task(
            run_copilot_welcome_message_job(app, copilot_welcome_job, template_name)
        )

        structlogger.info(
            "bot_builder_service.template_to_bot.success",
            files_generated=list(bot_files.keys()),
            copilot_template_prompt_job_id=copilot_template_prompt_job.id,
            copilot_welcome_job_id=copilot_welcome_job.id,
        )
        await push_job_status_event(
            job=job,
            status=JobStatus.done,
            payload={"copilot_welcome_job_id": copilot_welcome_job.id},
        )
        job_manager.mark_done(job)

    except TrainingError as exc:
        structlogger.debug(
            "template_to_bot_job.training_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.train_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except ValidationError as exc:
        # Log levels to include in the error message
        log_levels = ["error"]
        if config.VALIDATION_FAIL_ON_WARNINGS:
            log_levels.append("warning")

        structlogger.debug(
            "template_to_bot_job.validation_error",
            job_id=job.id,
            error=str(exc),
            all_validation_logs=exc.validation_logs,
            included_log_levels=log_levels,
        )
        error_message = exc.get_error_message_with_logs(log_levels=log_levels)
        await push_job_status_event(
            job, JobStatus.validation_error, message=error_message
        )
        job_manager.mark_done(job, error=error_message)

    except ProjectGenerationError as exc:
        structlogger.debug(
            "template_to_bot_job.generation_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.generation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except Exception as exc:
        # Capture full traceback
        structlogger.exception(
            "template_to_bot_job.unexpected_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


async def run_replace_all_files_job(
    app: "Sanic",
    job: JobInfo,
    bot_files: Dict[str, Any],
) -> None:
    """Run the replace-all-files job in the background.

    This replaces all bot files with the provided files and deletes any files
    not included in the request (excluding .rasa/ and models/ directories).

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        bot_files: Dictionary of file names to content for replacement.
    """
    project_generator = app.ctx.project_generator
    await push_job_status_event(job, JobStatus.received)

    try:
        project_generator.replace_all_bot_files(bot_files)

        # Validating
        await push_job_status_event(job, JobStatus.validating)
        training_input = project_generator.get_training_input()
        validation_error = await validate_project(training_input.importer)
        if validation_error:
            raise ValidationError(validation_error)
        await push_job_status_event(job, JobStatus.validation_success)

        # Training
        await push_job_status_event(job, JobStatus.training)
        agent = await train_and_load_agent(training_input)
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        # Send final done event with copilot training success response job ID
        copilot_training_success_job = job_manager.create_job()
        app.add_task(
            run_copilot_training_success_job(app, copilot_training_success_job)
        )

        await push_job_status_event(
            job=job,
            status=JobStatus.done,
            payload={
                "copilot_training_success_job_id": copilot_training_success_job.id
            },
        )
        job_manager.mark_done(job)

    except ValidationError as exc:
        log_levels = ["error"]
        if config.VALIDATION_FAIL_ON_WARNINGS:
            log_levels.append("warning")
        structlogger.debug(
            "replace_all_files_job.validation_error",
            job_id=job.id,
            error=str(exc),
            validation_logs=exc.validation_logs,
            included_log_levels=log_levels,
        )
        error_message = exc.get_error_message_with_logs(log_levels=log_levels)
        # Push error event and start copilot analysis job
        await push_error_and_start_copilot_analysis(
            app,
            job,
            JobStatus.validation_error,
            error_message,
            bot_files,
        )

        # After error mark job as done
        job_manager.mark_done(job, error=error_message)

    except TrainingError as exc:
        error_message = str(exc)
        structlogger.debug(
            "replace_all_files_job.train_error",
            job_id=job.id,
            error=error_message,
        )
        # Push error event and start copilot analysis job
        await push_error_and_start_copilot_analysis(
            app,
            job,
            JobStatus.train_error,
            error_message,
            bot_files,
        )

        # After error mark job as done
        job_manager.mark_done(job, error=error_message)

    except Exception as exc:
        # Capture full traceback for anything truly unexpected
        error_message = str(exc)
        structlogger.exception(
            "replace_all_files_job.unexpected_error",
            job_id=job.id,
            error=error_message,
        )

        # Push error event and start copilot analysis job
        await push_error_and_start_copilot_analysis(
            app,
            job,
            JobStatus.error,
            error_message,
            bot_files,
        )

        # After error mark job as done
        job_manager.mark_done(job, error=str(exc))


async def push_error_and_start_copilot_analysis(
    app: "Sanic",
    original_job: JobInfo,
    original_job_status: JobStatus,
    error_message: str,
    bot_files: Dict[str, Any],
) -> None:
    """Start a copilot analysis job and notify the client.

    Creates a copilot analysis job and sends the new job ID to the client. The new
    job runs in the background.

    Args:
        app: The Sanic application instance
        original_job: The original job that failed
        original_job_status: The status of the job that failed
        error_message: The error message to analyze
        bot_files: The bot files to include in analysis
    """
    # Create a copilot analysis job. Send the new job ID to the client and
    # run the Copilot Analysis job in the background.
    message = "Failed to train the assistant. Starting copilot analysis."

    copilot_job = job_manager.create_job()
    # Push the error status event for the original job
    await push_job_status_event(
        original_job,
        original_job_status,
        message=message,
        payload={"copilot_job_id": copilot_job.id},
    )
    # Run the copilot analysis job in the background
    app.add_task(
        run_copilot_training_error_analysis_job(
            app, copilot_job, error_message, bot_files
        )
    )
    structlogger.debug(
        f"update_files_job.{original_job_status.value}.copilot_analysis_start",
        event_info=message,
        job_id=original_job.id,
        error=error_message,
        copilot_job_id=copilot_job.id,
    )


async def run_copilot_training_error_analysis_job(
    app: "Sanic",
    job: JobInfo,
    training_error_message: str,
    bot_files: Dict[str, Any],
) -> None:
    """Run copilot training error analysis job."""
    await push_job_status_event(job, JobStatus.received)

    try:
        # Create message content blocks with log content and available files
        log_content_block = LogContent(
            type="log", content=training_error_message, context="training_error"
        )
        file_content_blocks = [
            FileContent(type="file", file_path=file_path, file_content=file_content)
            for file_path, file_content in bot_files.items()
        ]
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

        # Generate copilot response
        copilot_client = llm_service.instantiate_copilot()
        (
            original_stream,
            generation_context,
        ) = await copilot_client.generate_response(context)

        copilot_response_handler = llm_service.instantiate_handler(
            config.COPILOT_HANDLER_ROLLING_BUFFER_SIZE
        )
        intercepted_stream = copilot_response_handler.handle_response(original_stream)

        # Stream the copilot response as job events
        async for token in intercepted_stream:
            # Send each token as a job event using the same format as /copilot endpoint
            await push_job_status_event(
                job, JobStatus.copilot_analyzing, payload=token.sse_data
            )

        # Send references (if any) as part of copilot_analyzing stream
        if generation_context.relevant_documents:
            reference_section = copilot_response_handler.extract_references(
                generation_context.relevant_documents
            )
            await push_job_status_event(
                job, JobStatus.copilot_analyzing, payload=reference_section.sse_data
            )

        # Send original error log as part of copilot_analyzing stream
        training_error_log = TrainingErrorLog(logs=[log_content_block])
        await push_job_status_event(
            job, JobStatus.copilot_analyzing, payload=training_error_log.sse_data
        )

        # Persist the training error analysis to history
        full_text, _ = copilot_response_handler.extract_full_text_and_category()

        # Extract references if available
        references = None
        if generation_context.relevant_documents:
            reference_section = copilot_response_handler.extract_references(
                generation_context.relevant_documents
            )
            references = (
                reference_section.references if reference_section.references else None
            )

        await persist_training_error_analysis_to_history(
            text=full_text,
            logs=[log_content_block] if log_content_block else None,
            references=references,
            response_category=ResponseCategory.TRAINING_ERROR_LOG_ANALYSIS,
        )

        # Send success status
        await push_job_status_event(job, JobStatus.copilot_analysis_success)

        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

    except Exception as exc:
        structlogger.exception(
            "copilot_training_error_analysis_job.error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(
            job, JobStatus.copilot_analysis_error, message=str(exc)
        )
        job_manager.mark_done(job, error=str(exc))


async def run_copilot_welcome_message_job(
    app: "Sanic",
    job: JobInfo,
    template_name: Optional[ProjectTemplateName] = None,
) -> None:
    """Run the welcome message job in the background.

    This job sends a welcome message to the user after successful bot creation.
    For template-based bots, it sends a predefined message.
    For prompt-based bots, it can be extended to stream generated messages.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        template_name: The template name for template-based bots, None for prompt-based.
    """
    try:
        # Load welcome messages from YAML
        welcome_messages = load_copilot_welcome_messages()

        # Get the appropriate welcome message
        if template_name:
            welcome_message = welcome_messages.get(
                template_name.value,
                welcome_messages.get(PROMPT_TO_BOT_KEY),
            )
        else:
            welcome_message = welcome_messages.get(PROMPT_TO_BOT_KEY)

        # Send the welcome message as a single event
        await push_job_status_event(
            job,
            JobStatus.copilot_welcome_message,
            payload={
                "content": welcome_message,
                "response_category": "copilot",
                "completeness": "complete",
            },
        )

        # Persist the welcome message to conversation history
        await persist_copilot_message_to_history(text=welcome_message)

        # Mark job as done
        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

        structlogger.info(
            "copilot_welcome_message_job.success",
            job_id=job.id,
            template=template_name.value if template_name else PROMPT_TO_BOT_KEY,
        )

    except Exception as exc:
        structlogger.exception(
            "welcome_message_job.error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


async def run_copilot_training_success_job(
    app: "Sanic",
    job: JobInfo,
) -> None:
    """Run the training success job in the background.

    This job sends a training success message to the user after successful bot training.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
    """
    try:
        # Load copilot default messages from YAML
        internal_messages = load_copilot_handler_default_responses()

        # Get the appropriate training success message
        training_success_message = internal_messages.get("training_success_response")

        # Send the training success message
        await push_job_status_event(
            job,
            JobStatus.train_success_message,
            payload={
                "content": training_success_message,
                "response_category": "copilot",
                "completeness": "complete",
            },
        )

        # Send the training success category
        await push_job_status_event(
            job,
            JobStatus.train_success_message,
            payload={
                "response_category": "copilot_training_success",
                "completeness": "complete",
            },
        )

        # Persist the training success message to conversation history
        await persist_copilot_message_to_history(text=training_success_message)

        # Mark job as done
        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

        structlogger.info(
            "copilot_training_success_job.success",
            job_id=job.id,
        )

    except Exception as exc:
        structlogger.exception(
            "copilot_training_success_job.error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


async def run_copilot_template_prompt_job(
    app: "Sanic",
    job: JobInfo,
    template_name: ProjectTemplateName,
) -> None:
    """Run the template prompt job in the background.

    This job sends the template prompt as a user message at the start
    of a template-based bot creation.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        template_name: The template name to get the prompt for.
    """
    try:
        template_prompts = load_copilot_template_prompts()
        template_prompt = template_prompts.get(template_name.value)

        if not template_prompt:
            structlogger.warning(
                "copilot_template_prompt_job.no_prompt_found",
                job_id=job.id,
                template=template_name.value,
            )
            await push_job_status_event(job, JobStatus.done)
            job_manager.mark_done(job)
            return

        await push_job_status_event(
            job,
            JobStatus.copilot_template_prompt,
            payload={
                "content": template_prompt,
                "completeness": "complete",
            },
        )

        await persist_user_message_to_history(text=template_prompt)

        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

        structlogger.info(
            "copilot_template_prompt_job.success",
            job_id=job.id,
            template=template_name.value,
        )

    except Exception as exc:
        structlogger.exception(
            "copilot_template_prompt_job.error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


def _safe_tar_members(
    tar: tarfile.TarFile, destination_directory: Path
) -> List[tarfile.TarInfo]:
    """Get safe members for extraction to prevent path traversal and resource attacks.

    Args:
        tar: Open tar file handle
        destination_directory: Directory to which files will be extracted

    Returns:
        List of members that are safe to extract within destination_directory

    Raises:
        ProjectGenerationError: If archive violates security constraints
    """
    base_path = destination_directory.resolve()
    safe_members = []
    total_size = 0
    file_count = 0

    for member in tar.getmembers():
        name = member.name

        # Check file count limit
        file_count += 1
        if file_count > MAX_ARCHIVE_FILES:
            raise ProjectGenerationError(
                f"Archive contains too many files (>{MAX_ARCHIVE_FILES}).", attempts=1
            )

        # Skip empty names and absolute paths
        if not name or name.startswith("/") or name.startswith("\\"):
            continue

        # Disallow symlinks and hardlinks
        if member.issym() or member.islnk():
            continue

        # Check individual file size limit
        if member.size > MAX_ARCHIVE_FILE_SIZE:
            raise ProjectGenerationError(
                f"Archive contains file '{name}' that is too large "
                f"({member.size} bytes > {MAX_ARCHIVE_FILE_SIZE} bytes).",
                attempts=1,
            )

        # Check total size limit
        total_size += member.size
        if total_size > MAX_ARCHIVE_TOTAL_SIZE:
            raise ProjectGenerationError(
                "Archive total size too large "
                f"({total_size} bytes > {MAX_ARCHIVE_TOTAL_SIZE} bytes).",
                attempts=1,
            )

        # Compute the final path and ensure it's within base_path
        target_path = (base_path / name).resolve()
        try:
            target_path.relative_to(base_path)
        except ValueError:
            # Member would escape the destination directory
            continue

        safe_members.append(member)

    return safe_members


async def run_backup_to_bot_job(
    app: "Sanic",
    job: JobInfo,
    presigned_url: str,
) -> None:
    """Run the backup-to-bot job in the background.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        presigned_url: Presigned URL to download tar.gz backup data.
    """
    project_generator: ProjectGenerator = app.ctx.project_generator
    await push_job_status_event(job, JobStatus.received)

    temp_file_path = None
    try:
        # 1) Download and extract backup
        await push_job_status_event(job, JobStatus.generating)
        temp_file_path = await download_backup_from_url(presigned_url)

        # Clear existing project files, keeping .rasa and __pycache__
        project_path = Path(project_generator.project_folder)
        project_generator.cleanup(skip_files=[".rasa", "__pycache__"])

        # Extract the backup archive
        with tarfile.open(temp_file_path, "r:gz") as tar:
            safe_members = _safe_tar_members(tar, project_path)
            tar.extractall(path=project_path, members=safe_members)  # nosec B202:tarfile_unsafe_members

        await push_job_status_event(job, JobStatus.generation_success)

        # 2) Load existing model or train new one
        models_dir = project_path / "models"
        try:
            latest_model = get_local_model(str(models_dir))
        except ModelNotFound:
            latest_model = None

        if latest_model:
            # Load existing model
            structlogger.info(
                "backup_to_bot_job.loading_existing_model",
                job_id=job.id,
                model_path=latest_model,
            )
            await push_job_status_event(job, JobStatus.training)
            available_endpoints = Configuration.initialise_endpoints(
                endpoints_path=project_path / DEFAULT_ENDPOINTS_PATH
            ).endpoints
            agent = await load_agent(
                model_path=latest_model, endpoints=available_endpoints
            )
            update_agent(agent, app)
            await push_job_status_event(job, JobStatus.train_success)
        else:
            # Train new model
            await push_job_status_event(job, JobStatus.training)
            training_input = project_generator.get_training_input()
            agent = await train_and_load_agent(training_input)
            update_agent(agent, app)
            await push_job_status_event(job, JobStatus.train_success)

        # 3) Complete successfully
        bot_files = project_generator.get_bot_files()
        structlogger.info(
            "bot_builder_service.backup_to_bot.success",
            files_restored=list(bot_files.keys()),
            had_existing_model=bool(latest_model),
        )
        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

    except tarfile.ReadError as exc:
        raise ProjectGenerationError(
            f"Failed to extract backup archive: {exc}. "
            f"Please ensure the backup file is a valid tar.gz archive.",
            attempts=1,
        )
    except TrainingError as exc:
        structlogger.debug(
            "backup_to_bot_job.training_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.train_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except ValidationError as exc:
        log_levels = ["error"]
        if config.VALIDATION_FAIL_ON_WARNINGS:
            log_levels.append("warning")

        structlogger.debug(
            "backup_to_bot_job.validation_error",
            job_id=job.id,
            error=str(exc),
            all_validation_logs=exc.validation_logs,
            included_log_levels=log_levels,
        )
        error_message = exc.get_error_message_with_logs(log_levels=log_levels)
        await push_job_status_event(
            job, JobStatus.validation_error, message=error_message
        )
        job_manager.mark_done(job, error=error_message)

    except ProjectGenerationError as exc:
        structlogger.debug(
            "backup_to_bot_job.generation_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.generation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except Exception as exc:
        structlogger.exception(
            "backup_to_bot_job.unexpected_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))
    finally:
        # Always clean up temp file
        if temp_file_path:
            try:
                Path(temp_file_path).unlink(missing_ok=True)
            except Exception:
                pass
