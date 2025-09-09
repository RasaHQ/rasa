from typing import Any, Optional

import structlog
from sanic import Sanic

from rasa.builder import config
from rasa.builder.exceptions import (
    LLMGenerationError,
    ProjectGenerationError,
    TrainingError,
    ValidationError,
)
from rasa.builder.job_manager import JobInfo, job_manager
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
    job: JobInfo, status: JobStatus, message: Optional[str] = None
) -> None:
    event = JobStatusEvent.from_status(status=status.value, message=message)
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

        structlogger.info(
            "bot_builder_service.prompt_to_bot.success",
            files_generated=list(bot_files.keys()),
        )
        await push_job_status_event(job, JobStatus.done)
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

        # 3) Done
        structlogger.info(
            "bot_builder_service.template_to_bot.success",
            files_generated=list(bot_files.keys()),
        )
        await push_job_status_event(job, JobStatus.done)
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
    bot_files: dict,
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

        # 1. Validating
        await push_job_status_event(job, JobStatus.validating)
        training_input = project_generator.get_training_input()
        validation_error = await validate_project(training_input.importer)
        if validation_error:
            raise ValidationError(validation_error)
        await push_job_status_event(job, JobStatus.validation_success)

        # 2. Training
        await push_job_status_event(job, JobStatus.training)
        agent = await train_and_load_agent(training_input)
        update_agent(agent, app)
        await push_job_status_event(job, JobStatus.train_success)

        await push_job_status_event(job, JobStatus.done)
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
        await push_job_status_event(
            job, JobStatus.validation_error, message=error_message
        )
        job_manager.mark_done(job, error=error_message)

    except TrainingError as exc:
        structlogger.debug(
            "replace_all_files_job.train_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.train_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except Exception as exc:
        # Capture full traceback for anything truly unexpected
        structlogger.exception(
            "replace_all_files_job.unexpected_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))
