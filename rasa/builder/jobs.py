import asyncio
import subprocess
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
from rasa.builder.copilot import Copilot
from rasa.builder.copilot.constants import (
    PROMPT_TO_BOT_KEY,
    PROMPT_TO_BOT_TEMPLATE_KEY,
)
from rasa.builder.copilot.copilot_templated_message_provider import (
    copilot_handler_default_responses,
    copilot_template_prompts,
    copilot_welcome_messages,
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
from rasa.builder.git_service import GitOperationInProgressError
from rasa.builder.job_helpers import (
    handle_rollback_error,
    handle_rollback_validation_error,
    load_or_train_agent_for_commit,
    perform_rollback,
    push_error_and_start_copilot_analysis,
    push_job_status_event,
    train_and_load_and_link_agent,
)
from rasa.builder.job_manager import JobInfo, job_manager
from rasa.builder.models import (
    GitCommitInfo,
    JobStatus,
)
from rasa.builder.project_generator.project_generator import ProjectGenerator
from rasa.builder.telemetry.langfuse_compat import observe
from rasa.builder.telemetry.prompt_to_bot_langfuse_telemetry import (
    PromptToBotLangfuseTelemetry,
)
from rasa.builder.training_service import (
    try_load_existing_agent,
    update_agent,
)
from rasa.builder.validation_service import validate_project
from rasa.cli.scaffold import ProjectTemplateName

structlogger = structlog.get_logger()


async def send_heartbeat(job: JobInfo, interval: int = 15) -> None:
    """Send a heartbeat event every `interval` seconds while running."""
    try:
        while True:
            await asyncio.sleep(interval)
            await push_job_status_event(job, JobStatus.heartbeat)
    except asyncio.CancelledError:
        structlogger.debug("send_heartbeat.cancelled", job_id=job.id)
        raise


@observe(
    capture_input=False,
    capture_output=False,
)
async def run_prompt_to_bot_job(
    app: Any,
    job: JobInfo,
    prompt: str,
    user_id: Optional[str] = None,
) -> None:
    """Run the prompt-to-bot job in the background.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        prompt: The natural language prompt for bot generation.
        user_id: The user ID for the trace.
    """
    PromptToBotLangfuseTelemetry.setup_prompt_to_bot_trace(
        prompt=prompt,
        user_id=user_id,
        job_id=job.id,
    )
    project_generator: ProjectGenerator = app.ctx.project_generator

    copilot_template_prompt_job = job_manager.create_job()
    heartbeat_task = asyncio.create_task(send_heartbeat(job))

    try:
        # 0. Persist initial user prompt
        app.add_task(
            run_copilot_template_prompt_job(
                app, copilot_template_prompt_job, custom_prompt=prompt
            )
        )
        await push_job_status_event(
            job,
            JobStatus.received,
            payload={"copilot_template_prompt_job_id": copilot_template_prompt_job.id},
        )

        # 1. Generating
        await push_job_status_event(job, JobStatus.generating)
        (
            attempts,
            commit_sha,
        ) = await project_generator.generate_project_with_retries(
            prompt,
            template=ProjectTemplateName.BASIC,
        )
        bot_files = project_generator.get_bot_files()
        await push_job_status_event(job, JobStatus.generation_success)

        # Persist prompt to history after generation
        # This ensures the project directory and .rasa folder exist
        try:
            if prompt:
                await persist_user_message_to_history(text=prompt)
        except Exception as persist_exc:
            # Don't fail the job if persistence fails, just log it
            structlogger.error(
                "prompt_to_bot_job.persist_prompt_failed",
                job_id=job.id,
                prompt=prompt,
                error=str(persist_exc),
            )

        # 2. Training
        await push_job_status_event(job, JobStatus.training)
        agent = await train_and_load_and_link_agent(
            project_generator, commit_sha, role="copilot", action="generation"
        )
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        # 3. Create copilot welcome message job
        copilot_welcome_job = job_manager.create_job(commit_sha)
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

        PromptToBotLangfuseTelemetry.update_prompt_to_bot_trace_output_success(
            bot_files=bot_files,
            attempts=attempts,
            max_attempts=config.PROJECT_GENERATION_MAX_RETRIES,
        )

    except TrainingError as exc:
        structlogger.debug(
            "prompt_to_bot_job.training_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.train_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

        PromptToBotLangfuseTelemetry.update_prompt_to_bot_trace_output_failure(
            error=exc,
            attempts=config.PROJECT_GENERATION_MAX_RETRIES,
            max_attempts=config.PROJECT_GENERATION_MAX_RETRIES,
        )

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

        PromptToBotLangfuseTelemetry.update_prompt_to_bot_trace_output_failure(
            error=exc,
            attempts=config.PROJECT_GENERATION_MAX_RETRIES,
            max_attempts=config.PROJECT_GENERATION_MAX_RETRIES,
        )

    except (ProjectGenerationError, LLMGenerationError) as exc:
        structlogger.debug(
            "prompt_to_bot_job.generation_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.generation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

        PromptToBotLangfuseTelemetry.update_prompt_to_bot_trace_output_failure(
            error=exc,
            attempts=config.PROJECT_GENERATION_MAX_RETRIES,
            max_attempts=config.PROJECT_GENERATION_MAX_RETRIES,
        )

    except Exception as exc:
        # Capture full traceback
        structlogger.exception(
            "prompt_to_bot_job.unexpected_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

        PromptToBotLangfuseTelemetry.update_prompt_to_bot_trace_output_failure(
            error=exc,
            attempts=config.PROJECT_GENERATION_MAX_RETRIES,
            max_attempts=config.PROJECT_GENERATION_MAX_RETRIES,
        )
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            structlogger.warning(
                "prompt_to_bot_job.heartbeat_task_failed",
                job_id=job.id,
                error=str(exc),
            )


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
        commit_sha = await project_generator.init_from_template(template_name)
        bot_files = project_generator.get_bot_files()
        await push_job_status_event(job, JobStatus.generation_success)

        # Persist template prompt to history after template initialization
        # This ensures the project directory and .rasa folder exist
        try:
            template_prompt = copilot_template_prompts().get(template_name.value)
            if template_prompt:
                await persist_user_message_to_history(text=template_prompt)
        except Exception as persist_exc:
            # Don't fail the job if persistence fails, just log it
            structlogger.error(
                "template_to_bot_job.persist_template_prompt_failed",
                job_id=job.id,
                template=template_name.value,
                error=str(persist_exc),
            )

        await push_job_status_event(job, JobStatus.training)
        agent = await try_load_existing_agent(project_generator.project_folder)
        if agent is None:
            agent = await train_and_load_and_link_agent(project_generator, commit_sha)
        else:
            structlogger.info(
                "bot_builder_service.template_to_bot.agent_loaded_from_cache",
            )
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        copilot_welcome_job = job_manager.create_job(commit_sha)
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
    commit_info: GitCommitInfo,
) -> None:
    """Run the replace-all-files job in the background.

    This replaces all bot files with the provided files and deletes any files
    not included in the request (excluding .rasa/ and models/ directories).

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        bot_files: Dictionary of file names to content for replacement.
        commit_info: Optional arguments for the commit
    """
    project_generator: ProjectGenerator = app.ctx.project_generator
    await push_job_status_event(job, JobStatus.received)

    commit_sha: Optional[str] = None
    try:
        commit_sha = await project_generator.replace_all_bot_files(
            bot_files, commit_info
        )
        # Validating
        await push_job_status_event(job, JobStatus.validating)
        training_input = project_generator.get_training_input()
        validation_error = await validate_project(training_input.importer)
        if validation_error:
            raise ValidationError(validation_error)
        await push_job_status_event(job, JobStatus.validation_success)

        # Training
        await push_job_status_event(job, JobStatus.training)
        agent = await train_and_load_and_link_agent(project_generator, commit_sha)
        update_agent(agent, app)

        await push_job_status_event(job, JobStatus.train_success)

        # Send final done event with copilot training success response job ID
        copilot_training_success_job = job_manager.create_job(commit_sha=commit_sha)
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

    except GitOperationInProgressError as exc:
        error_message = str(exc)
        structlogger.debug(
            "replace_all_files_job.operation_in_progress",
            job_id=job.id,
            error=error_message,
        )
        await push_job_status_event(job, JobStatus.error, message=error_message)
        job_manager.mark_done(job, error=error_message)

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
            commit_sha,
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
            commit_sha,
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
            commit_sha,
        )

        # After error mark job as done
        job_manager.mark_done(job, error=str(exc))


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
        copilot_client = Copilot()
        (
            copilot_response_handler,
            generation_context,
        ) = await copilot_client.generate_response(context)

        commit_info = None
        if job.commit_sha:
            # Get commit info
            commit_info = await app.ctx.project_generator.git_service.get_commit_info(
                job.commit_sha
            )

            commit_info["training_success"] = False

            # Send commit
            await push_job_status_event(
                job,
                JobStatus.copilot_analyzing,
                payload={
                    "response_category": "copilot",
                    "commit": commit_info,
                    "completeness": "complete",
                },
            )

        # Send original error log as part of copilot_analyzing stream
        training_error_log = TrainingErrorLog(logs=[log_content_block])
        await push_job_status_event(
            job, JobStatus.copilot_analyzing, payload=training_error_log.sse_data
        )

        # Stream the copilot response as job events
        async for token in copilot_response_handler.stream():
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

        # Persist the training error analysis to history
        full_text = copilot_response_handler.extract_full_text()

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
            commit=commit_info,
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
        welcome_messages = copilot_welcome_messages()

        # Get the appropriate welcome message
        if template_name:
            welcome_message = welcome_messages.get(
                template_name.value,
                welcome_messages.get(PROMPT_TO_BOT_KEY),
            )
        else:
            welcome_message = await app.ctx.project_generator.generate_welcome_message(
                default_welcome_message=welcome_messages.get(PROMPT_TO_BOT_KEY),
                template_welcome_message=welcome_messages.get(
                    PROMPT_TO_BOT_TEMPLATE_KEY
                ),
            )

        commit_info = None
        if job.commit_sha:
            # Get commit info
            commit_info = await app.ctx.project_generator.git_service.get_commit_info(
                job.commit_sha
            )

            commit_info["training_success"] = True

            # Send commit
            await push_job_status_event(
                job,
                JobStatus.copilot_welcome_message,
                payload={
                    "response_category": "copilot",
                    "commit": commit_info,
                    "completeness": "complete",
                },
            )

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
        await persist_copilot_message_to_history(
            text=welcome_message, commit=commit_info
        )

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
        internal_messages = copilot_handler_default_responses()

        # Get the appropriate training success message
        training_success_message = internal_messages.get("training_success_response")

        commit_info = None
        if job.commit_sha:
            # Get commit info
            commit_info = await app.ctx.project_generator.git_service.get_commit_info(
                job.commit_sha
            )

            # Add training success
            commit_info["training_success"] = True

            # Send the training success message
            await push_job_status_event(
                job,
                JobStatus.train_success_message,
                payload={
                    "response_category": "copilot",
                    "commit": commit_info,
                    "completeness": "complete",
                },
            )

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

        await persist_copilot_message_to_history(
            text=training_success_message, commit=commit_info
        )

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


async def run_copilot_rollback_message_job(
    app: "Sanic",
    job: JobInfo,
    internal_message_key: str,
    status: JobStatus,
    training_error_log: Optional[TrainingErrorLog] = None,
) -> None:
    """Run the rollback message job in the background.

    This job sends a rollback message to the user after rollback.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        internal_message_key: The key of the internal message to send.
        status: The status of the job.
        training_error_log: Optional training error log to include in the message.

    Returns:
        None
    """
    try:
        # Load copilot default messages from YAML
        internal_messages = copilot_handler_default_responses()

        # Get the appropriate rollback message
        rollback_message = internal_messages.get(internal_message_key)

        commit_info = None
        if job.commit_sha:
            # Get commit info
            commit_info = await app.ctx.project_generator.git_service.get_commit_info(
                job.commit_sha
            )

            if training_error_log:
                # Send error log
                await push_job_status_event(
                    job, status, payload=training_error_log.sse_data
                )

                commit_info["training_success"] = False

            else:
                commit_info["training_success"] = True

            # Send commit
            await push_job_status_event(
                job,
                status,
                payload={
                    "response_category": "copilot",
                    "commit": commit_info,
                    "completeness": "complete",
                },
            )

        # Send the message
        await push_job_status_event(
            job,
            status,
            payload={
                "content": rollback_message,
                "response_category": "copilot",
                "completeness": "complete",
            },
        )

        await persist_copilot_message_to_history(
            text=rollback_message,
            commit=commit_info,
            logs=training_error_log.logs if training_error_log else None,
        )

        # Mark job as done
        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

        structlogger.info(
            "copilot_go_back_in_time_success_job.success",
            job_id=job.id,
        )

    except Exception as exc:
        structlogger.exception(
            "copilot_go_back_in_time_success_job.error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


async def run_change_branch_job(
    app: "Sanic",
    job: JobInfo,
    branch_name: str,
    create_if_not_exists: bool = False,
) -> None:
    """Run the change branch job in the background.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        branch_name: The branch name to checkout.
        create_if_not_exists: Whether to create the branch if it doesn't exist.
    """
    project_generator: ProjectGenerator = app.ctx.project_generator

    await push_job_status_event(job, JobStatus.received)

    try:
        # 1. Switch branch
        await push_job_status_event(job, JobStatus.switching_branch)
        commit_sha = await project_generator.checkout_branch(
            branch_name, create_if_not_exists
        )
        await push_job_status_event(job, JobStatus.branch_switch_success)

        # 2. Training
        await push_job_status_event(job, JobStatus.training)
        agent = await load_or_train_agent_for_commit(project_generator, job, commit_sha)
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        # 3. Done
        structlogger.info(
            "bot_builder_service.change_branch.success",
            branch_name=branch_name,
        )
        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

    except GitOperationInProgressError as exc:
        structlogger.debug(
            "change_branch_job.operation_in_progress",
            job_id=job.id,
            error=str(exc),
            branch_name=branch_name,
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except subprocess.CalledProcessError as exc:
        structlogger.debug(
            "change_branch_job.branch_switch_error",
            job_id=job.id,
            error=str(exc),
            branch_name=branch_name,
        )
        await push_job_status_event(
            job, JobStatus.branch_switch_error, message=str(exc)
        )
        job_manager.mark_done(job, error=str(exc))

    except TrainingError as exc:
        structlogger.debug(
            "change_branch_job.training_error",
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
            "change_branch_job.validation_error",
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

    except Exception as exc:
        # Capture full traceback
        structlogger.exception(
            "change_branch_job.unexpected_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))


async def run_rollback_job(
    app: "Sanic",
    job: JobInfo,
    commit_sha: str,
) -> None:
    """Run rollback job following existing patterns.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        commit_sha: SHA of the commit to rollback to.
    """
    project_generator: ProjectGenerator = app.ctx.project_generator
    await push_job_status_event(job, JobStatus.received)

    try:
        # 1. Rolling back
        rollback_commit_sha = await perform_rollback(project_generator, job, commit_sha)

        structlogger.info(
            "bot_builder_service.rollback.success", commit_sha=rollback_commit_sha
        )
        # 2. Validating (ok if fails)
        await push_job_status_event(job, JobStatus.validating)
        training_input = project_generator.get_training_input()
        validation_error = await validate_project(training_input.importer)
        if validation_error:
            raise ValidationError(validation_error)
        await push_job_status_event(job, JobStatus.validation_success)
        structlogger.info(
            "bot_builder_service.rollback.validation_success",
            commit_sha=rollback_commit_sha,
        )
        # 3. Training
        # we use the prior commit sha, since that saves us from a retrain in
        # case we have a trained model for the prior commit. using the
        # rollback sha wouldn't make sense as that is a new commit, so there
        # would surely no model be trained for that
        agent = await load_or_train_agent_for_commit(project_generator, job, commit_sha)

        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)
        structlogger.info(
            "bot_builder_service.rollback.train_success", commit_sha=rollback_commit_sha
        )

        # 4. Send rollback message
        copilot_rollback_message_job = job_manager.create_job(
            commit_sha=rollback_commit_sha
        )
        app.add_task(
            run_copilot_rollback_message_job(
                app,
                copilot_rollback_message_job,
                "rollback_message_response",
                JobStatus.rollback_message,
            )
        )

        # 5. Done
        await push_job_status_event(
            job,
            JobStatus.done,
            payload={
                "copilot_rollback_message_job_id": copilot_rollback_message_job.id
            },
        )
        job_manager.mark_done(job)

    except GitOperationInProgressError as exc:
        await handle_rollback_error(job, exc, commit_sha, JobStatus.error)
    except subprocess.CalledProcessError as exc:
        await handle_rollback_error(job, exc, commit_sha, JobStatus.rollback_error)
    except ValidationError as exc:
        await handle_rollback_validation_error(app, job, exc, rollback_commit_sha)
    except TrainingError as exc:
        await handle_rollback_error(job, exc, commit_sha, JobStatus.train_error)
    except Exception as exc:
        await handle_rollback_error(
            job, exc, commit_sha, JobStatus.error, log_traceback=True
        )


async def run_copilot_template_prompt_job(
    app: "Sanic",
    job: JobInfo,
    template_name: Optional[ProjectTemplateName] = None,
    custom_prompt: Optional[str] = None,
) -> None:
    """Run the template prompt job in the background.

    This job sends the template prompt as a user message to the frontend
    at the start of a template-based bot creation. The message is persisted
    to history separately after template initialization.

    Args:
        app: The Sanic application instance.
        job: The job information instance.
        template_name: Optional template name to get the prompt for.
        custom_prompt: Optional custom prompt by user.
    """
    try:
        template_prompts = copilot_template_prompts()
        prompt = custom_prompt or (
            template_prompts.get(template_name.value) if template_name else None
        )

        if not prompt:
            structlogger.warning(
                "copilot_template_prompt_job.no_prompt_found",
                job_id=job.id,
                template=template_name.value if template_name else None,
                custom_prompt=custom_prompt if custom_prompt else None,
            )
            await push_job_status_event(job, JobStatus.done)
            job_manager.mark_done(job)
            return

        await push_job_status_event(
            job,
            JobStatus.copilot_template_prompt,
            payload={
                "content": prompt,
                "completeness": "complete",
            },
        )

        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

        structlogger.info(
            "copilot_template_prompt_job.success",
            job_id=job.id,
            template_name=template_name.value if template_name else None,
            custom_prompt=custom_prompt if custom_prompt else None,
            prompt=prompt,
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

        # Ensure Git is initialized
        project_generator.migrate_git_repository_if_needed()
        commit_sha = await project_generator.git_service.get_current_commit_sha()
        agent = await load_or_train_agent_for_commit(project_generator, job, commit_sha)

        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        structlogger.info(
            "bot_builder_service.backup_to_bot.success",
            commit_sha=commit_sha,
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
