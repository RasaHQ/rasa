"""Helper functions for job execution.

This module contains utility functions used by various job implementations
to reduce duplication and improve maintainability.
"""

from typing import Any, Dict, Optional

import structlog
from sanic import Sanic

from rasa.builder.git_service import link_model_to_commit
from rasa.builder.job_manager import JobInfo, job_manager
from rasa.builder.models import JobStatus, JobStatusEvent
from rasa.builder.project_generator import ProjectGenerator
from rasa.builder.training_service import train_and_load_agent
from rasa.core.agent import Agent

structlogger = structlog.get_logger()


async def push_job_status_event(
    job: JobInfo,
    status: JobStatus,
    message: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Push a job status event to the job queue.

    Args:
        job: The job information instance.
        status: The job status to push.
        message: Optional message to include with the status.
        payload: Optional payload data to include with the status.
    """
    event = JobStatusEvent.from_status(
        status=status.value, message=message, payload=payload
    )
    job.status = status.value
    await job.put(event)


async def push_error_and_start_copilot_analysis(
    app: "Sanic",
    original_job: JobInfo,
    original_job_status: JobStatus,
    error_message: str,
    bot_files: Dict[str, Any],
    commit_sha: Optional[str] = None,
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
        commit_sha: The commit SHA to link the copilot analysis job to
    """
    # Import here to avoid circular dependency
    from rasa.builder.jobs import run_copilot_training_error_analysis_job

    # Create a copilot analysis job. Send the new job ID to the client and
    # run the Copilot Analysis job in the background.
    message = "Failed to train the assistant. Starting copilot analysis."

    copilot_job = job_manager.create_job(commit_sha)
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


# Rollback-specific helpers


async def perform_rollback(
    project_generator: ProjectGenerator, job: JobInfo, commit_sha: str
) -> str:
    """Perform the git rollback operation.

    Args:
        project_generator: The project generator instance
        job: The job information instance
        commit_sha: SHA of the commit to rollback to

    Returns:
        The SHA of the rollback commit
    """
    await push_job_status_event(job, JobStatus.rolling_back)
    rollback_commit_sha = await project_generator.git_service.rollback_to_commit(
        commit_sha
    )
    await push_job_status_event(job, JobStatus.rollback_success)
    return rollback_commit_sha


async def train_and_load_and_link_agent(
    project_generator: ProjectGenerator, commit_sha: str
) -> Agent:
    """Train and load an agent and link it to a commit."""
    agent = await train_and_load_agent(project_generator.get_training_input())
    await link_model_to_commit(project_generator.git_service, agent, commit_sha)
    return agent


async def load_or_train_agent_for_commit(
    project_generator: ProjectGenerator, job: JobInfo, commit_sha: str
) -> Agent:
    """Load existing model for commit or retrain if missing.

    Args:
        project_generator: The project generator instance
        job: The job information instance
        commit_sha: SHA of the commit

    Returns:
        The loaded or trained agent
    """
    await push_job_status_event(job, JobStatus.training)

    agent = await project_generator.git_service.load_model_for_commit(commit_sha)
    if agent is None:
        # Retrain if model missing
        return await train_and_load_and_link_agent(project_generator, commit_sha)

    return agent


async def handle_rollback_error(
    job: JobInfo,
    exc: Exception,
    commit_sha: str,
    error_status: JobStatus,
    log_traceback: bool = False,
) -> None:
    """Handle rollback job errors with appropriate logging and status updates.

    Args:
        job: The job information instance
        exc: The exception that occurred
        commit_sha: SHA of the commit being rolled back to
        error_status: The error status to set
        log_traceback: Whether to log the full traceback
    """
    log_method = structlogger.exception if log_traceback else structlogger.debug
    log_method(
        f"rollback_job.{error_status.value}",
        job_id=job.id,
        error=str(exc),
        commit_sha=commit_sha,
    )
    await push_job_status_event(job, error_status, message=str(exc))
    job_manager.mark_done(job, error=str(exc))


# Revert-specific helpers


async def perform_revert(
    project_generator: ProjectGenerator, job: JobInfo, commit_sha: str
) -> str:
    """Perform the git revert operation."""
    await push_job_status_event(job, JobStatus.reverting)
    revert_commit_sha = await project_generator.git_service.revert_to_commit(commit_sha)
    await push_job_status_event(job, JobStatus.revert_success)
    return revert_commit_sha


async def handle_revert_error(
    job: JobInfo,
    exc: Exception,
    commit_sha: str,
    error_status: JobStatus,
    log_traceback: bool = False,
) -> None:
    """Handle revert job errors with appropriate logging and status updates."""
    log_method = structlogger.exception if log_traceback else structlogger.debug
    log_method(
        f"revert_job.{error_status.value}",
        job_id=job.id,
        error=str(exc),
        commit_sha=commit_sha,
    )
    await push_job_status_event(job, error_status, message=str(exc))
    job_manager.mark_done(job, error=str(exc))
