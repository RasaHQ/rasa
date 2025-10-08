from typing import Any, Dict, Optional

import structlog
from sanic import Sanic

from rasa.builder import config
from rasa.builder.copilot.constants import (
    PROMPT_TO_BOT_KEY,
)
from rasa.builder.copilot.copilot_templated_message_provider import (
    load_copilot_handler_default_responses,
    load_copilot_welcome_messages,
)
from rasa.builder.copilot.models import (
    CopilotContext,
    FileContent,
    InternalCopilotRequestChatMessage,
    LogContent,
    ResponseCategory,
    TrainingErrorLog,
)
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

    await push_job_status_event(job, JobStatus.received)

    try:
        # 1) Generating
        await push_job_status_event(job, JobStatus.generating)
        await project_generator.init_from_template(template_name)
        bot_files = project_generator.get_bot_files()
        await push_job_status_event(job, JobStatus.generation_success)

        # 2) Training
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

        # 3) Create copilot welcome message job
        copilot_welcome_job = job_manager.create_job()
        app.add_task(
            run_copilot_welcome_message_job(app, copilot_welcome_job, template_name)
        )

        # 4) Done - include welcome job ID in payload
        structlogger.info(
            "bot_builder_service.template_to_bot.success",
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
