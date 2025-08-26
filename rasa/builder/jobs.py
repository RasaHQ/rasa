from typing import Any, Optional

import structlog
from sanic import Sanic

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
from rasa.builder.training_service import train_and_load_agent
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
    project_generator = app.ctx.project_generator
    input_channel = app.ctx.input_channel

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
        importer = project_generator._create_importer()
        app.ctx.agent = await train_and_load_agent(importer)
        input_channel.agent = app.ctx.agent
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
        structlogger.debug(
            "prompt_to_bot_job.validation_error", job_id=job.id, error=str(exc)
        )
        await push_job_status_event(job, JobStatus.validation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

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
    project_generator = app.ctx.project_generator
    input_channel = app.ctx.input_channel

    await push_job_status_event(job, JobStatus.received)

    try:
        # 1) Generating
        await push_job_status_event(job, JobStatus.generating)
        await project_generator.init_from_template(template_name)
        bot_files = project_generator.get_bot_files()
        await push_job_status_event(job, JobStatus.generation_success)

        # 2) Training
        await push_job_status_event(job, JobStatus.training)
        importer = project_generator._create_importer()
        app.ctx.agent = await train_and_load_agent(importer)
        input_channel.agent = app.ctx.agent
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
        structlogger.debug(
            "template_to_bot_job.validation_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.validation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

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


async def run_update_files_job(
    app: "Sanic",
    job: JobInfo,
    bot_files: dict,
) -> None:
    project_generator = app.ctx.project_generator
    input_channel = app.ctx.input_channel
    await push_job_status_event(job, JobStatus.received)

    try:
        project_generator.update_bot_files(bot_files)

        # 1. Validating
        await push_job_status_event(job, JobStatus.validating)
        importer = project_generator._create_importer()
        validation_error = await validate_project(importer)
        if validation_error:
            raise ValidationError(validation_error)
        await push_job_status_event(job, JobStatus.validation_success)

        # 2. Training
        await push_job_status_event(job, JobStatus.training)
        app.ctx.agent = await train_and_load_agent(importer)
        input_channel.agent = app.ctx.agent
        await push_job_status_event(job, JobStatus.train_success)

        await push_job_status_event(job, JobStatus.done)
        job_manager.mark_done(job)

    except ValidationError as exc:
        structlogger.debug(
            "update_files_job.validation_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.validation_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except TrainingError as exc:
        structlogger.debug(
            "update_files_job.train_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.train_error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))

    except Exception as exc:
        # Capture full traceback for anything truly unexpected
        structlogger.exception(
            "update_files_job.unexpected_error",
            job_id=job.id,
            error=str(exc),
        )
        await push_job_status_event(job, JobStatus.error, message=str(exc))
        job_manager.mark_done(job, error=str(exc))
