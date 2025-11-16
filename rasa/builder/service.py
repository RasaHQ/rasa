# mypy: disable-error-code=misc
import asyncio
import sys
import time
from http import HTTPStatus
from typing import Any, Optional

import langfuse
import structlog
from sanic import Blueprint, HTTPResponse, response
from sanic.request import Request
from sanic_openapi import openapi

import rasa
from rasa.builder.auth import HEADER_USER_ID, is_auth_required_now, protected
from rasa.builder.config import (
    COPILOT_ASSISTANT_TRACKER_MAX_TURNS,
    COPILOT_HANDLER_ROLLING_BUFFER_SIZE,
    GUARDRAILS_ENABLE_BLOCKING,
    HELLO_RASA_PROJECT_ID,
    LAKERA_ASSISTANT_HISTORY_GUARDRAIL_PROJECT_ID,
    LAKERA_COPILOT_HISTORY_GUARDRAIL_PROJECT_ID,
)
from rasa.builder.copilot.constants import DEFAULT_COPILOT_CHAT_ID, ROLE_USER
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.exceptions import CopilotStreamError
from rasa.builder.copilot.history_store import persist_copilot_message_to_history
from rasa.builder.copilot.models import (
    ConversationKey,
    CopilotChatMessage,
    CopilotContext,
    CopilotHistoryResponse,
    CopilotTurnRequest,
    GeneratedContent,
    ReferenceEntry,
    ReferenceSection,
    ResponseCategory,
    ResponseCompleteness,
    TextContent,
)
from rasa.builder.download import create_bot_project_archive
from rasa.builder.guardrails.constants import (
    BLOCK_SCOPE_PROJECT,
    BLOCK_SCOPE_USER,
    BlockScope,
)
from rasa.builder.guardrails.store import guardrails_store
from rasa.builder.job_manager import job_manager
from rasa.builder.jobs import (
    run_backup_to_bot_job,
    run_prompt_to_bot_job,
    run_replace_all_files_job,
    run_template_to_bot_job,
)
from rasa.builder.llm_service import llm_service
from rasa.builder.logging_utils import (
    capture_exception_with_context,
    get_recent_logs,
)
from rasa.builder.models import (
    AgentStatus,
    ApiErrorResponse,
    AssistantInfo,
    BotData,
    BotFiles,
    JobCreateResponse,
    JobStatus,
    JobStatusEvent,
    PromptRequest,
    RestoreFromBackupRequest,
    ServerSentEvent,
    TemplateRequest,
)
from rasa.builder.project_generator import ProjectGenerator
from rasa.builder.shared.tracker_context import TrackerContext
from rasa.builder.telemetry.copilot_langfuse_telemetry import CopilotLangfuseTelemetry
from rasa.builder.telemetry.copilot_segment_telemetry import CopilotSegmentTelemetry
from rasa.core.agent import Agent
from rasa.core.channels.studio_chat import StudioChatInput
from rasa.core.exceptions import AgentNotReady
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.flows.yaml_flows_io import get_flows_as_json
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.utils import DOMAIN_KEYS
from rasa.utils.json_utils import extract_values
from rasa.utils.openapi import model_to_schema

structlogger = structlog.get_logger()

# Create the blueprint
bp = Blueprint("bot_builder", url_prefix="/api")


def setup_project_generator(project_folder: str) -> ProjectGenerator:
    """Initialize and return a ProjectGenerator instance."""
    # Ensure the project folder is in sys.path
    if project_folder not in sys.path:
        sys.path.insert(0, project_folder)

    structlogger.info(
        "bot_builder_service.service_initialized", project_folder=project_folder
    )

    return ProjectGenerator(project_folder)


def get_project_generator(request: Request) -> ProjectGenerator:
    """Get the project generator from app context."""
    return request.app.ctx.project_generator


def get_input_channel(request: Request) -> StudioChatInput:
    """Get the input channel from app context."""
    return request.app.ctx.input_channel


async def extract_bot_data_from_agent(agent: Agent) -> BotData:
    """Extract BotData from an Agent.

    Args:
        agent: The agent to extract data from

    Returns:
        BotData containing flows, domain, config, endpoints, and nlu data
    """
    domain = agent.domain.as_dict() if agent.domain else {}
    flows = (
        await agent.processor.get_flows()
        if agent.processor
        else FlowsList(underlying_flows=[])
    )
    return BotData(
        flows=get_flows_as_json(flows),
        domain=extract_values(domain, DOMAIN_KEYS),
    )


async def get_agent_status(request: Request) -> AgentStatus:
    """Get the status of the agent."""
    if request.app.ctx.agent is None:
        return AgentStatus.not_loaded
    agent: Agent = request.app.ctx.agent
    if agent.is_ready():
        return AgentStatus.ready
    return AgentStatus.not_ready


# Health check endpoint
@bp.route("/", methods=["GET"])
@openapi.summary("Health check endpoint")
@openapi.description(
    "Returns the health status of the Bot Builder service including version "
    "information and authentication requirements"
)
@openapi.tag("health")
@openapi.response(
    200,
    {
        "application/json": {
            "status": str,
            "service": str,
            "rasa_version": str,
            "auth_required": bool,
            "agent_status": str,
        }
    },
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
async def health(request: Request) -> HTTPResponse:
    """Health check endpoint."""
    project_generator = get_project_generator(request)
    return response.json(
        {
            "status": "ok",
            "service": "bot-builder",
            "rasa_version": rasa.__version__,
            "agent_status": await get_agent_status(request),
            "auth_required": is_auth_required_now(
                project_info=project_generator.project_info
            ),
        }
    )


@bp.route("/job-events/<job_id>", methods=["GET"])
@openapi.summary("Stream job progress events")
@openapi.description(
    "Stream server-sent events (SSE) tracking real-time job progress.\n\n"
    "**Connect with:** `Accept: text/event-stream`.\n\n"
    "**SSE Event Example:**\n"
    "```text\n"
    "event: received\n"
    'data: {"status": "received"}\n'
    "```\n\n"
)
@openapi.tag("job-events")
@openapi.parameter(
    "job_id", str, location="path", description="The id of the job to stream events for"
)
@openapi.response(
    200,
    {"text/event-stream": str},
    description="Server-sent events stream. See documentation for event types.",
    example=(
        "event: received\n"
        'data: {"status": "received"}\n'
        "\n"
        "event: generating\n"
        'data: {"status": "generating"}\n'
    ),
)
@openapi.response(
    404,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Unknown job_id: No such job exists.",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def job_events(request: Request, job_id: str) -> HTTPResponse:
    try:
        job = job_manager.get_job(job_id)
        if job is None:
            return response.json(
                ApiErrorResponse(
                    error="Job not found", details={"job_id": job_id}
                ).model_dump(),
                status=404,
            )

        stream = await request.respond(content_type="text/event-stream")

        try:
            async for evt in job.event_stream():
                await stream.send(evt.format())
        except Exception as exc:
            # Handle exceptions within the SSE stream context
            capture_exception_with_context(
                exc,
                "bot_builder_service.job_events.streaming_error",
                extra={"job_id": job_id},
                tags={"endpoint": "/api/job-events/<job_id>"},
            )
            # Send error event in SSE format instead of JSON response
            error_event = JobStatusEvent.from_status(
                status=JobStatus.error.value,
                message=f"Failed to stream job events: {exc}",
            ).format()
            await stream.send(error_event)
        finally:
            await stream.eof()

        return stream
    except Exception as exc:
        # This exception handler only applies before stream.respond() is called
        capture_exception_with_context(
            exc,
            "bot_builder_service.job_events.unexpected_error",
            extra={"job_id": job_id},
            tags={"endpoint": "/api/job-events/<job_id>"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to stream job events", details={"error": str(exc)}
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/prompt-to-bot", methods=["POST"])
@openapi.summary("Generate bot from natural language prompt")
@openapi.description(
    "Creates a complete conversational AI bot from a natural language prompt. "
    "Returns immediately with a job ID. Connect to `/job-events/<job_id>` to "
    "receive server-sent events (SSE) for real-time progress tracking "
    "throughout the bot creation process.\n\n"
    "**SSE Event Flow** (via `/job-events/<job_id>`):\n"
    "1. `received` - Request received by server\n"
    "2. `generating` - Generating bot project files\n"
    "3. `generation_success` - Bot generation completed successfully\n"
    "4. `training` - Training the bot model\n"
    "5. `train_success` - Model training completed\n"
    "6. `done` - Bot creation completed\n\n"
    "**Error Events:**\n"
    "- `generation_error` - Failed to generate bot from prompt\n"
    "- `train_error` - Bot generated but training failed\n"
    "- `validation_error` - Generated bot configuration is invalid\n"
    "- `error` - Unexpected error occurred\n\n"
    "**Usage:**\n"
    "1. Send POST request with Content-Type: application/json\n"
    "2. The response will be a JSON object `{job_id: ...}`\n"
    "3. Connect to `/job-events/<job_id>` for a server-sent event stream of progress."
)
@openapi.tag("bot-generation")
@openapi.body(
    {"application/json": model_to_schema(PromptRequest)},
    description="Prompt request with natural language description.",
    required=True,
    example={
        "prompt": (
            "Create a customer support bot that can help users with order inquiries, "
            "product questions, and returns processing. The bot should be friendly "
            "and able to escalate to human agents when needed."
        )
    },
)
@openapi.response(
    200,
    {"application/json": model_to_schema(JobCreateResponse)},
    description="Job created. Poll or subscribe to /job-events/<job_id> for progress.",
)
@openapi.response(
    400,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Validation error in request payload",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def handle_prompt_to_bot(request: Request) -> HTTPResponse:
    """Handle prompt-to-bot generation requests."""
    try:
        payload = PromptRequest(**request.json)
    except Exception as exc:
        return response.json(
            ApiErrorResponse(
                error="Invalid request", details={"error": str(exc)}
            ).model_dump(),
            status=400,
        )

    try:
        # Allocate job and schedule background task
        job = job_manager.create_job()
        request.app.add_task(run_prompt_to_bot_job(request.app, job, payload.prompt))
        return response.json(JobCreateResponse(job_id=job.id).model_dump(), status=200)
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.prompt_to_bot.unexpected_error",
            tags={"endpoint": "/api/prompt-to-bot"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to create prompt-to-bot job",
                details={"error": str(exc)},
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/template-to-bot", methods=["POST"])
@openapi.summary("Generate bot from predefined template")
@openapi.description(
    "Creates a complete conversational AI bot from a predefined template. "
    "Returns immediately with a job ID. Connect to `/job-events/<job_id>` to "
    "receive server-sent events (SSE) for real-time progress tracking "
    "throughout the bot creation process.\n\n"
    "**SSE Event Flow** (via `/job-events/<job_id>`):\n"
    "1. `received` - Request received by server\n"
    "2. `generating` - Initializing bot from template\n"
    "3. `generation_success` - Template initialization completed successfully\n"
    "4. `training` - Training the bot model\n"
    "5. `train_success` - Model training completed\n"
    "6. `done` - Bot creation completed\n\n"
    "**Error Events:**\n"
    "- `generation_error` - Failed to initialize bot from template\n"
    "- `train_error` - Template loaded but training failed\n"
    "- `validation_error` - Template configuration is invalid\n"
    "- `error` - Unexpected error occurred\n\n"
    "**Usage:**\n"
    "1. Send POST request with Content-Type: application/json\n"
    "2. The response will be a JSON object `{job_id: ...}`\n"
    "3. Connect to `/job-events/<job_id>` for a server-sent event stream of progress."
)
@openapi.tag("bot-generation")
@openapi.body(
    {"application/json": model_to_schema(TemplateRequest)},
    description="Template request with template name.",
    required=True,
    example={"template_name": "telco"},
)
@openapi.response(
    200,
    {"application/json": model_to_schema(JobCreateResponse)},
    description="Job created. Poll or subscribe to /job-events/<job_id> for progress.",
)
@openapi.response(
    400,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Validation error in request payload or invalid template name",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def handle_template_to_bot(request: Request) -> HTTPResponse:
    """Create a new template-to-bot job and return job_id immediately."""
    try:
        template_data = TemplateRequest(**request.json)
    except Exception as exc:
        return response.json(
            ApiErrorResponse(
                error="Invalid request", details={"error": str(exc)}
            ).model_dump(),
            status=400,
        )

    try:
        # allocate job and schedule background task
        job = job_manager.create_job()
        request.app.add_task(
            run_template_to_bot_job(request.app, job, template_data.template_name)
        )
        return response.json(JobCreateResponse(job_id=job.id).model_dump(), status=200)
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.template_to_bot.unexpected_error",
            tags={"endpoint": "/api/template-to-bot"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to create template-to-bot job",
                details={"error": str(exc)},
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/backup-to-bot", methods=["POST"])
@openapi.summary("Generate bot from backup archive")
@openapi.description(
    "Creates a complete conversational AI bot from a backup tar.gz archive. "
    "Returns immediately with a job ID. Connect to `/job-events/<job_id>` to "
    "receive server-sent events (SSE) for real-time progress tracking "
    "throughout the bot restoration process.\n\n"
    "**SSE Event Flow** (via `/job-events/<job_id>`):\n"
    "1. `received` - Request received by server\n"
    "2. `generating` - Extracting and restoring bot from backup\n"
    "3. `generation_success` - Backup restoration completed successfully\n"
    "4. `training` - Training the bot model (if no existing model found)\n"
    "5. `train_success` - Model training completed (if training was needed)\n"
    "6. `done` - Bot restoration completed\n\n"
    "**Error Events:**\n"
    "- `generation_error` - Failed to restore bot from backup\n"
    "- `train_error` - Backup restored but training failed\n"
    "- `validation_error` - Restored bot configuration is invalid\n"
    "- `error` - Unexpected error occurred\n\n"
    "**Usage:**\n"
    "1. Send POST request with Content-Type: application/json\n"
    "2. The response will be a JSON object `{job_id: ...}`\n"
    "3. Connect to `/job-events/<job_id>` for a server-sent event stream of progress."
)
@openapi.tag("bot-generation")
@openapi.body(
    {"application/json": model_to_schema(RestoreFromBackupRequest)},
    description="Backup request with presigned URL to tar.gz archive.",
    required=True,
    example={"presigned_url": "https://s3.amazonaws.com/bucket/path?signature=..."},
)
@openapi.response(
    200,
    {"application/json": model_to_schema(JobCreateResponse)},
    description="Job created. Poll or subscribe to /job-events/<job_id> for progress.",
)
@openapi.response(
    400,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Validation error in request payload or invalid presigned URL",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def handle_backup_to_bot(request: Request) -> HTTPResponse:
    """Handle backup-to-bot restoration requests."""
    try:
        payload = RestoreFromBackupRequest(**request.json)
    except Exception as exc:
        return response.json(
            ApiErrorResponse(
                error="Invalid request", details={"error": str(exc)}
            ).model_dump(),
            status=400,
        )

    try:
        # Allocate job and schedule background task
        job = job_manager.create_job()
        request.app.add_task(
            run_backup_to_bot_job(request.app, job, payload.presigned_url)
        )
        return response.json(JobCreateResponse(job_id=job.id).model_dump(), status=200)
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.backup_to_bot.unexpected_error",
            tags={"endpoint": "/api/backup-to-bot"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to create backup-to-bot job",
                details={"error": str(exc)},
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/files", methods=["GET"])
@openapi.summary("Get bot files")
@openapi.description(
    "Retrieves the current bot configuration files including domain.yml, "
    "config.yml, flows.yml, NLU data, and other project files as a "
    "dictionary mapping file names to their string contents"
)
@openapi.tag("bot-files")
@openapi.response(
    200,
    {"application/json": {str: Optional[str]}},
    description="Bot files retrieved successfully",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def get_bot_files(request: Request) -> HTTPResponse:
    """Get current bot files."""
    try:
        project_generator = get_project_generator(request)
        bot_files = project_generator.get_bot_files()
        return response.json(bot_files)
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.get_bot_files.unexpected_error",
            tags={"endpoint": "/api/files", "method": "GET"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to retrieve bot files", details={"error": str(exc)}
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/files", methods=["POST"])
@openapi.summary("Replace all bot files")
@openapi.description(
    "Replaces all bot configuration files with the provided files, deletes any "
    "files not included in the request (excluding .rasa/ and models/ directories), "
    "and retrains the model. Returns immediately with a job ID. Connect to "
    "`/job-events/<job_id>` for real-time SSE progress tracking."
    "\n\n"
    "**File Management:**\n"
    "- All files in the request are written to the project folder\n"
    "- Files not included in the request are deleted from the project\n"
    "- Files/folders starting with `.rasa/` or `models/` are excluded from deletion\n\n"
    "**SSE Event Flow:** (available via /job-events/<job_id>)\n"
    "1. `received` - Request received by server\n"
    "2. `validating` - Validating bot configuration files\n"
    "3. `validation_success` - File validation completed successfully\n"
    "4. `training` - Training the bot model with updated files\n"
    "5. `train_success` - Model training completed\n"
    "6. `done` - Bot files replacement completed\n\n"
    "**Error Events (can occur at any time):**\n"
    "- `validation_error` - Bot configuration files are invalid\n"
    "- `train_error` - Files updated but training failed\n"
    "- `copilot_analysis_start` - Copilot analysis started (includes `copilot_job_id` "
    "in payload)\n"
    "- `error` - Unexpected error occurred\n\n"
    "**Copilot Analysis Events:**\n"
    "When training or validation fails, a separate copilot analysis job is "
    "automatically started. The `copilot_analysis_start` event includes a "
    "`copilot_job_id` in the payload.\nConnect to `/job-events/<copilot_job_id>` to "
    "receive the following events:\n"
    "- `copilot_analyzing` - Copilot is analyzing errors and providing suggestions. "
    "Uses the same SSE event payload format as the `/copilot` endpoint with `content`, "
    "`response_category`, and `completeness` fields.\n"
    "- `copilot_analysis_success` - Copilot analysis completed with references.\n"
    "- `copilot_analysis_error` - Copilot analysis failed\n\n"
    "**Usage:**\n"
    "1. Send POST request with Content-Type: application/json\n"
    "2. The response will be a JSON object `{job_id: ...}`\n"
    "3. Connect to `/job-events/<job_id>` for a server-sent event stream of progress."
)
@openapi.tag("bot-files")
@openapi.body(
    {"application/json": {str: Optional[str]}},
    description=(
        "A dictionary mapping file names to their complete content. "
        "All files in the project will be replaced with these files. "
        "Files not included in the request will be deleted from the project "
        "(except for .rasa/ and models/ directories). "
        "The file name should be the relative path from the project root."
    ),
    required=True,
    example={
        "domain.yml": (
            "version: '3.1'\n"
            "intents:\n  - greet\n  - goodbye\n"
            "responses:\n  utter_greet:\n  - text: 'Hello!'\n"
            "  utter_goodbye:\n  - text: 'Goodbye!'"
        ),
        "config.yml": (
            "version: '3.1'\n"
            "pipeline:\n  - name: WhitespaceTokenizer\n"
            "  - name: RegexFeaturizer\n  - name: LexicalSyntacticFeaturizer\n"
            "  - name: CountVectorsFeaturizer\n"
            "policies:\n  - name: MemoizationPolicy\n  - name: RulePolicy"
        ),
        "data/nlu.yml": (
            "version: '3.1'\n"
            "nlu:\n- intent: greet\n  examples: |\n    - hello\n    - hi"
        ),
    },
)
@openapi.response(
    200,
    {"application/json": model_to_schema(JobCreateResponse)},
    description=(
        "Job created. Poll or subscribe to /job-events/<job_id> "
        "for progress and SSE updates on file replacement and training."
    ),
)
@openapi.response(
    400,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Validation error in bot files",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.response(
    401,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description=(
        "Authentication failed - Authorization header missing or invalid. "
        "Auth may be conditionally required depending on server "
        "configuration."
    ),
)
@openapi.parameter(
    "Authorization",
    description=(
        "Bearer token for authentication. Required after auth start window "
        "or if configured."
    ),
    _in="header",
    required=False,
    schema=str,
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
@protected()
async def replace_all_bot_files(request: Request) -> HTTPResponse:
    """Replace all bot files with server-sent events for progress tracking."""
    try:
        bot_files = request.json
    except Exception as exc:
        return response.json(
            ApiErrorResponse(
                error="Invalid request", details={"error": str(exc)}
            ).model_dump(),
            status=400,
        )

    try:
        job = job_manager.create_job()
        request.app.add_task(run_replace_all_files_job(request.app, job, bot_files))
        return response.json(JobCreateResponse(job_id=job.id).model_dump(), status=200)
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.replace_all_bot_files.unexpected_error",
            tags={"endpoint": "/api/files", "method": "POST"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to replace bot files", details={"error": str(exc)}
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/data", methods=["GET"])
@openapi.summary("Get bot data")
@openapi.description(
    "Retrieves the current bot data in CALM import format with flows, domain, "
    "config, endpoints, and NLU data"
)
@openapi.tag("bot-info")
@openapi.response(
    200,
    {"application/json": model_to_schema(BotData)},
    description="Bot data retrieved successfully",
)
@openapi.response(
    409,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Agent not ready",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def get_bot_data(request: Request) -> HTTPResponse:
    """Get current bot data in CALM import format."""
    try:
        agent: Optional[Agent] = request.app.ctx.agent
        if not agent:
            raise AgentNotReady(
                "Can't retrieve the data without an agent being loaded."
            )

        bot_data = await extract_bot_data_from_agent(agent)

        return response.json(bot_data.model_dump())
    except AgentNotReady as e:
        return response.json(
            ApiErrorResponse(
                error="Agent not ready",
                details={"error": str(e)},
            ).model_dump(),
            status=HTTPStatus.CONFLICT,
        )
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.get_bot_data.unexpected_error",
            tags={"endpoint": "/api/data"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to retrieve bot data",
                details={"error": str(exc)},
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/assistant", methods=["GET"])
@openapi.summary("Get assistant info")
@openapi.description(
    "Returns basic information about the loaded assistant, including the assistant id "
    "as configured in the model's metadata (from config.yml)."
)
@openapi.tag("bot-info")
@openapi.response(
    200,
    {"application/json": model_to_schema(AssistantInfo)},
    description="Assistant info retrieved successfully",
)
@openapi.response(
    409,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Agent not ready",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def get_bot_info(request: Request) -> HTTPResponse:
    """Return assistant info including assistant id from model metadata."""
    try:
        agent: Optional[Agent] = request.app.ctx.agent
        if not agent:
            raise AgentNotReady("Can't retrieve bot info without a loaded agent.")

        assistant_id: Optional[str] = (
            agent.processor.model_metadata.assistant_id
            if agent.processor
            and agent.processor.model_metadata
            and hasattr(agent.processor.model_metadata, "assistant_id")
            else None
        )

        return response.json(AssistantInfo(assistant_id=assistant_id).model_dump())
    except AgentNotReady as e:
        return response.json(
            ApiErrorResponse(
                error="Agent not ready",
                details={"error": str(e)},
            ).model_dump(),
            status=HTTPStatus.CONFLICT,
        )
    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.get_bot_info.unexpected_error",
            tags={"endpoint": "/api/assistant"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to retrieve bot info",
                details={"error": str(exc)},
            ).model_dump(),
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )


@bp.route("/download", methods=["GET"])
@openapi.summary("Download bot project as tar.gz")
@openapi.description(
    "Downloads the current bot project files as a compressed tar.gz archive. "
    "Includes all configuration files and a .env file with RASA_PRO_LICENSE."
)
@openapi.tag("bot-files")
@openapi.parameter(
    "exclude_models_directory",
    bool,
    location="query",
    description="Whether to exclude the models directory",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
@openapi.parameter(
    "project_name",
    description="Name of the project for the archive filename and pyproject.toml",
    _in="query",
    required=False,
    schema=str,
)
@openapi.response(
    200,
    {"application/gzip": bytes},
    description="Bot project downloaded successfully as tar.gz",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error",
)
async def download_bot_project(request: Request) -> HTTPResponse:
    """Download bot project as tar.gz archive."""
    try:
        # Get query parameters
        exclude_models_directory = (
            request.args.get("exclude_models_directory", "true").lower() == "true"
        )
        project_name = request.args.get("project_name", "bot-project")

        # Get bot files
        project_generator = get_project_generator(request)
        bot_files = project_generator.get_bot_files(
            exclude_models_directory=exclude_models_directory
        )

        # Create tar.gz archive
        tar_data = create_bot_project_archive(
            bot_files, project_name, project_generator.project_folder
        )

        structlogger.info(
            "bot_builder_service.download_bot_project.success",
            user_sub=(getattr(request.ctx, "auth_payload", None) or {}).get("sub"),
            files_count=len(bot_files),
            archive_size=len(tar_data),
            payload=getattr(request.ctx, "auth_payload", None),
            project_name=project_name,
        )

        return response.raw(
            tar_data,
            content_type="application/gzip",
            headers={
                "Content-Disposition": f"attachment; filename={project_name}.tar.gz"
            },
        )

    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.download_bot_project.unexpected_error",
            tags={"endpoint": "/api/download"},
        )
        return response.json(
            ApiErrorResponse(
                error="Failed to create bot project archive",
                details={"error": str(exc)},
            ).model_dump(),
            status=500,
        )


@bp.route("/copilot", methods=["POST"])
@openapi.summary("AI copilot for bot building")
@openapi.description(
    "Provides LLM-powered copilot assistance with streaming markdown responses. "
    "Returns server-sent events (SSE) for real-time streaming of copilot responses.\n\n"
    "The event's `event` field is the type of event. For this endpoint it will always "
    "be: `copilot_response`.\n\n"
)
@openapi.tag("copilot")
@openapi.body(
    {"application/json": model_to_schema(CopilotTurnRequest)},
    description=(
        "Copilot request containing: "
        "1. a single user message, "
        "2. session ID for tracking conversation context with the bot being built, "
        "3. optional chat ID for copilot conversation (defaults to 'default'), "
        "4. optional project ID for conversation isolation, "
        "Conversation history is stored and managed on the server."
    ),
    required=True,
)
@openapi.response(
    200,
    {"text/event-stream": model_to_schema(ServerSentEvent)},
    description=(
        "Server-sent events stream with copilot responses and references. "
        "The event's `event` field is the type "
        "of event. For this endpoint it will always be: `copilot_response`. "
        "The event's data field is a JSON dump. The following describes the event data "
        "field:\n"
        "- `response_category` can be one of the following:\n"
        "  - `copilot` - Stream token generated by the copilot.\n"
        "  - `out_of_scope_detection` - Response coming from the the out of scope detection.\n"  # noqa: E501
        "  - `roleplay_detection` - Response coming from the the roleplay detection.\n"
        "  - `reference` - Reference section.\n"
        "  - `training_error_log_analysis` - Used to flag the responses that are training error log analysis.\n"  # noqa: E501
        "  - `e2e_testing_error_log_analysis` - Used to flag the responses that are e2e testing error log analysis.\n"  # noqa: E501
        "  - `guardrail_policy_violation` - Used to flag the responses that are flagged as a violation of the guardrail policy.\n\n"  # noqa: E501
        "- `completeness`: Whether this is a streaming token or complete response.\n\n"
        "- `content`: The actual response content (for streaming tokens).\n\n"
        "- `references`: (Only for `reference` response category) List of reference entries. Each reference entry is a dictionary with the following keys:\n\n"  # noqa: E501
        "   - `index`: Reference index as an integer number.\n"
        "   - `title`: Reference title as a string.\n"
        "   - `url`: Reference URL as a string.\n\n"
    ),
    examples={
        "Streaming token from copilot": {
            "summary": "Streaming token from copilot",
            "value": GeneratedContent(
                content="<token generated by the copilot>",
                response_category=ResponseCategory.COPILOT,
                response_completeness=ResponseCompleteness.TOKEN,
            )
            .to_sse_event()
            .model_dump(),
        },
        "Complete response for the following response categories: out_of_scope_detection / roleplay_detection / training_error_log_analysis / e2e_testing_error_log_analysis / guardrail_policy_violation": {  # noqa: E501
            "summary": "Complete response from copilot",
            "value": GeneratedContent(
                content="<response from the out of scope detection>",
                response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
                response_completeness=ResponseCompleteness.COMPLETE,
            )
            .to_sse_event()
            .model_dump(),
        },
        "Response with the references": {
            "summary": "Reference section with all references",
            "value": ReferenceSection(
                references=[
                    ReferenceEntry(
                        index=1,
                        title="Title of the reference",
                        url="https://rasa.com/docs/...",
                    ),
                    ReferenceEntry(
                        index=2,
                        title="Title of another reference",
                        url="https://rasa.com/docs/...",
                    ),
                ],
                response_category=ResponseCategory.REFERENCE,
                response_completeness=ResponseCompleteness.COMPLETE,
            )
            .to_sse_event()
            .model_dump(),
        },
    },
)
@openapi.response(
    400,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Validation error in request",
)
@openapi.response(
    502,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="LLM generation failed.",
)
@openapi.response(
    500,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Internal server error.",
)
@openapi.response(
    401,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description=(
        "Authentication failed - Authorization header missing or invalid. "
        "Auth may be conditionally required depending on server "
        "configuration."
    ),
)
@openapi.parameter(
    "Authorization",
    description=(
        "Bearer token for authentication. Required after auth start window "
        "or if configured."
    ),
    _in="header",
    required=False,
    schema=str,
)
@openapi.parameter(
    HEADER_USER_ID,
    description=("Required user id used for telemetry and guardrails (X-User-Id)."),
    _in="header",
    required=True,
    schema=str,
)
@protected()
# Disable automatic input/output capture for langfuse tracing
# This allows manual control over what data is sent to langfuse
@langfuse.observe(capture_input=False, capture_output=False)
async def copilot(request: Request) -> None:
    """Handle copilot requests with streaming markdown responses."""
    sse = await request.respond(content_type="text/event-stream")
    project_generator = get_project_generator(request)

    try:
        # 1. Validate and unpack input
        req = CopilotTurnRequest(**request.json)

        # Require user identifier via header and fail fast if missing
        user_id = (request.headers.get(HEADER_USER_ID) or "").strip()
        if not user_id:
            structlogger.error(
                "bot_builder_service.copilot.missing_or_empty_user_id_header",
                event_info=f"Missing or empty required header: {HEADER_USER_ID}",
            )
            await sse.send(
                ServerSentEvent(
                    event="error",
                    data={
                        "error": f"Missing or empty required header: {HEADER_USER_ID}"
                    },
                ).format()
            )
            return

        telemetry = CopilotSegmentTelemetry(
            project_id=HELLO_RASA_PROJECT_ID, user_id=user_id
        )
        structlogger.debug("builder.copilot.telemetry.request.init")

        # TODO: This can be removed once Langfuse is completed.
        if req.message and req.message.role == ROLE_USER:
            structlogger.debug("builder.copilot.telemetry.request.user_turn")
            # Offload telemetry logging to a background task
            request.app.add_task(
                asyncio.to_thread(
                    telemetry.log_user_turn,
                    req.message.get_flattened_text_content(),
                )
            )

        # 2. Check if we need to block the request due to too many guardrails violations
        if (scope := await _get_copilot_block_scope(user_id)) is not None:
            message = CopilotResponseHandler.respond_to_guardrail_blocked(scope)
            await sse.send(message.to_sse_event().format())
            return

        # 3. Load existing server-side history and build candidate turn
        # Use chat_id from payload if provided, otherwise use default
        chat_id = (req.chat_id or "").strip() or DEFAULT_COPILOT_CHAT_ID
        conversation_key = ConversationKey(chat_id=chat_id)
        existing_history = await llm_service.history_store.get(conversation_key)

        # Build the full history for context (existing + new user message)
        candidate_history = [*existing_history, req.message]

        # 4. Get the necessary context for the copilot
        tracker_context = await get_tracker_context_for_copilot(request, req, user_id)
        relevant_assistant_files = get_relevant_assistant_files_for_copilot(
            project_generator,
        )
        context = CopilotContext(
            tracker_context=tracker_context,
            assistant_logs=get_recent_logs(),
            assistant_files=relevant_assistant_files,
            copilot_chat_history=candidate_history,
        )

        # Persist the user message to history
        try:
            await llm_service.history_store.append(conversation_key, req.message)
        except Exception as exc:
            structlogger.error(
                "builder.copilot.history.user_message_persist_failed", error=str(exc)
            )

        # 5. Run guardrail policy checks. If any policy violations are detected,
        #    send a response and end the stream.
        guardrail_response: Optional[GeneratedContent] = None
        if llm_service.guardrails_policy_checker is not None:
            guardrail_response = await llm_service.guardrails_policy_checker.check_copilot_chat_for_policy_violations(  # noqa: E501
                context=context,
                hello_rasa_user_id=user_id,
                hello_rasa_project_id=HELLO_RASA_PROJECT_ID,
                lakera_project_id=LAKERA_COPILOT_HISTORY_GUARDRAIL_PROJECT_ID,
            )
        if guardrail_response is not None:
            blocked_or_violation_message = (
                await _handle_guardrail_violation_and_maybe_block(
                    sse=sse,
                    user_id=user_id,
                    violation_response=guardrail_response,
                )
            )

            # Persist the guardrail response as well
            guardrail_chat_message = CopilotChatMessage(
                role="copilot",
                content=[
                    TextContent(type="text", text=blocked_or_violation_message.content)
                ],
                response_category=blocked_or_violation_message.response_category,
            )
            try:
                await llm_service.history_store.append(
                    conversation_key, guardrail_chat_message
                )
            except Exception as exc:
                structlogger.error(
                    "builder.copilot.history.guardrail_response_persist_failed",
                    error=str(exc),
                )
            return

        # 6. Get the original response stream from copilot and handle it with the
        #    copilot response handler
        start_timestamp = time.perf_counter()
        copilot_client = llm_service.instantiate_copilot()
        (original_stream, generation_context) = await copilot_client.generate_response(
            context
        )

        copilot_response_handler = llm_service.instantiate_handler(
            COPILOT_HANDLER_ROLLING_BUFFER_SIZE
        )
        intercepted_stream = copilot_response_handler.handle_response(original_stream)

        # 7. Stream the intercepted response
        async for token in intercepted_stream:
            await sse.send(token.to_sse_event().format())

        # 8a. Offload metabase telemetry logging to a background task
        usage_stats = copilot_client.usage_statistics
        request.app.add_task(
            asyncio.to_thread(
                telemetry.log_copilot_from_handler,
                handler=copilot_response_handler,
                used_documents=generation_context.relevant_documents,
                latency_ms=int((time.perf_counter() - start_timestamp) * 1000),
                system_message=generation_context.system_message,
                chat_history=generation_context.chat_history,
                last_user_message=(
                    req.message.get_flattened_text_content()
                    if (req.message and req.message.role == ROLE_USER)
                    else None
                ),
                tracker_event_attachments=generation_context.tracker_event_attachments,
                model=usage_stats.model or "N/A",
                prompt_tokens=usage_stats.prompt_tokens or 0,
                completion_tokens=usage_stats.completion_tokens or 0,
                total_tokens=usage_stats.total_tokens or 0,
                cached_prompt_tokens=(
                    copilot_client.usage_statistics.cached_prompt_tokens or 0
                ),
            )
        )
        # 8b. Setup output trace attributes for Langfuse
        CopilotLangfuseTelemetry.setup_copilot_endpoint_call_trace_attributes(
            hello_rasa_project_id=HELLO_RASA_PROJECT_ID or "N/A",
            chat_id=req.session_id or "N/A",
            user_id=user_id,
            request=req,
            handler=copilot_response_handler,
            relevant_documents=generation_context.relevant_documents,
            copilot_context=context,
        )

        # 9. Once the stream is over, extract and send references
        #    if any documents were used
        reference_section = None
        if generation_context.relevant_documents:
            reference_section = copilot_response_handler.extract_references(
                generation_context.relevant_documents
            )
            await sse.send(reference_section.to_sse_event().format())

        # 10. Append final assistant message to server-side history
        full_text, category = copilot_response_handler.extract_full_text_and_category()
        if full_text:
            try:
                # Pass references directly if they exist
                references = reference_section.references if reference_section else None
                await persist_copilot_message_to_history(
                    text=full_text,
                    chat_id=chat_id,
                    response_category=category,
                    references=references,
                )
            except Exception as exc:
                structlogger.error(
                    "builder.copilot.history.persist_failed", error=str(exc)
                )
        else:
            structlogger.warning(
                "builder.copilot.history.no_assistant_text",
                session_id=req.session_id,
            )

    except CopilotStreamError as e:
        capture_exception_with_context(
            e,
            "bot_builder_service.copilot.generation_error",
            extra={"session_id": req.session_id},
            tags={"endpoint": "/api/copilot"},
        )
        await sse.send(
            ServerSentEvent(
                event="error",
                data={"error": str(e)},
            ).format()
        )

    except Exception as exc:
        capture_exception_with_context(
            exc,
            "bot_builder_service.copilot.unexpected_error",
            extra={"session_id": req.session_id if "req" in locals() else None},
            tags={"endpoint": "/api/copilot"},
        )
        await sse.send(
            ServerSentEvent(
                event="error",
                data={"error": str(exc)},
            ).format()
        )

    finally:
        await sse.eof()


@bp.route("/copilot/internal_message_templates/<template_name>", methods=["GET"])
@openapi.summary("Get templated response for copilot internal message")
@openapi.description(
    "Returns the templated response text for a given template name from the "
    "copilot internal message formatter. This endpoint provides access to the "
    "predefined templates used for formatting internal system messages."
)
@openapi.tag("copilot")
@openapi.parameter(
    "template_name",
    str,
    location="path",
    description=(
        "The template name to get the template for."
        "(e.g., 'training_error_log_analysis', 'e2e_testing_error_log_analysis')",
    ),
)
@openapi.response(
    200,
    {"application/json": {"template": str, "template_name": str}},
    description="Successfully retrieved the template for the given template name",
    example={
        "template": "The assistant training failed. Your task is ...",
        "template_name": "training_error_log_analysis",
    },
)
@openapi.response(
    404,
    {"application/json": model_to_schema(ApiErrorResponse)},
    description="Template not found for the given template name",
)
@openapi.parameter(
    HEADER_USER_ID,
    description=(
        "Optional user id to associate requests (e.g., for telemetry/guardrails)."
    ),
    _in="header",
    required=False,
    schema=str,
)
async def get_copilot_internal_message_template(
    request: Request, template_name: str
) -> HTTPResponse:
    """Get templated response for copilot internal message formatter."""
    try:
        # Try to get the template for the given template name
        template = llm_service.copilot_internal_message_templates.get(template_name)
        structlogger.info(
            "bot_builder_service.get_copilot_internal_message_template.template_found",
            template_name=template_name,
            template=template,
        )

        if template is None:
            structlogger.warning(
                "bot_builder_service.get_copilot_internal_message_template.not_found",
                template_name=template_name,
            )
            return response.json(
                ApiErrorResponse(
                    error="Template not found", details={"template_name": template_name}
                ).model_dump(),
                status=404,
            )

        return response.json({"template": template, "template_name": template_name})

    except Exception as e:
        structlogger.error(
            "bot_builder_service.get_copilot_internal_message_template.error",
            error=str(e),
            template_name=template_name,
        )
        return response.json(
            ApiErrorResponse(
                error="Internal server error", details={"template_name": template_name}
            ).model_dump(),
            status=500,
        )


@bp.route("/copilot/history", methods=["GET"])
@openapi.summary("Get Copilot chat history")
@openapi.description("Return stored copilot chat history.")
@openapi.tag("copilot")
@openapi.parameter(
    "chat_id",
    description="Optional chat ID to get history for (defaults to 'default').",
    _in="query",
    required=False,
    schema=str,
)
@openapi.response(
    200,
    {"application/json": model_to_schema(CopilotHistoryResponse)},
    description="Chat history retrieved successfully",
)
async def get_copilot_history(request: Request) -> HTTPResponse:
    """Get copilot chat history."""
    # Use chat_id from query parameter if provided, otherwise use default
    chat_id = (request.args.get("chat_id") or "").strip() or DEFAULT_COPILOT_CHAT_ID
    conversation_key = ConversationKey(chat_id=chat_id)
    messages = await llm_service.history_store.get(conversation_key)
    return response.json(CopilotHistoryResponse(messages=messages).model_dump())


@bp.route("/copilot/history", methods=["DELETE"])
@openapi.summary("Delete Copilot chat history")
@openapi.description("Delete stored copilot chat history.")
@openapi.tag("copilot")
@openapi.parameter(
    "chat_id",
    description="Optional chat ID to delete history for (defaults to 'default').",
    _in="query",
    required=False,
    schema=str,
)
@openapi.response(
    200,
    {
        "application/json": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
            },
        }
    },
    description="History deleted successfully",
)
async def delete_copilot_history(request: Request) -> HTTPResponse:
    """Delete copilot chat history."""
    # Use chat_id from query parameter if provided, otherwise use default
    chat_id = (request.args.get("chat_id") or "").strip() or DEFAULT_COPILOT_CHAT_ID
    conversation_key = ConversationKey(chat_id=chat_id)
    await llm_service.history_store.delete(conversation_key)
    return response.json({"status": "deleted"})


async def current_tracker_from_input_channel(
    app: Any, session_id: str
) -> Optional[DialogueStateTracker]:
    """Generate chat bot context from current conversation."""
    if app.ctx.agent and session_id:
        return await app.ctx.agent.tracker_store.retrieve(session_id)
    else:
        return None


async def _get_copilot_block_scope(user_id: str) -> Optional[BlockScope]:
    """Return the guardrail block scope for Copilot if blocked.

    Args:
        user_id: User identifier.

    Returns:
        'user' or 'project' when blocked, otherwise None.
    """
    if not GUARDRAILS_ENABLE_BLOCKING:
        return None

    return await guardrails_store.check_block_scope(user_id)


async def _handle_guardrail_violation_and_maybe_block(
    sse: Any,
    violation_response: GeneratedContent,
    user_id: str,
) -> GeneratedContent:
    """Record a violation, apply block if threshold crossed, and respond.

    Args:
        sse: Active SSE stream.
        violation_response: The default violation warning response.
        user_id: User identifier.

    Returns:
        The GeneratedContent message that was sent to the client.
    """
    if not GUARDRAILS_ENABLE_BLOCKING:
        await sse.send(violation_response.to_sse_event().format())
        return violation_response

    result = await guardrails_store.record_violation(user_id)

    if result.user_blocked_now:
        message = CopilotResponseHandler.respond_to_guardrail_blocked(BLOCK_SCOPE_USER)
    elif result.project_blocked_now:
        message = CopilotResponseHandler.respond_to_guardrail_blocked(
            BLOCK_SCOPE_PROJECT
        )
    else:
        message = violation_response

    await sse.send(message.to_sse_event().format())
    return message


@langfuse.observe(capture_input=False, capture_output=False)
async def get_tracker_context_for_copilot(
    request: Request,
    req: CopilotTurnRequest,
    user_id: str,
) -> Optional[TrackerContext]:
    """Check the assistant chat for guardrail policy violations.

    Args:
        request: The request object.
        req: The CopilotTurnRequest object.
        user_id: The user ID.

    Returns:
        The tracker context if the tracker is available.
    """
    tracker = await current_tracker_from_input_channel(request.app, req.session_id)
    tracker_context = TrackerContext.from_tracker(
        tracker, max_turns=COPILOT_ASSISTANT_TRACKER_MAX_TURNS
    )
    if (
        tracker_context is not None
        and llm_service.guardrails_policy_checker is not None
    ):
        tracker_context = await llm_service.guardrails_policy_checker.check_assistant_chat_for_policy_violations(  # noqa: E501
            tracker_context=tracker_context,
            hello_rasa_user_id=user_id,
            hello_rasa_project_id=HELLO_RASA_PROJECT_ID,
            lakera_project_id=LAKERA_ASSISTANT_HISTORY_GUARDRAIL_PROJECT_ID,
        )

    # Track the retrieved tracker context
    CopilotLangfuseTelemetry.trace_copilot_tracker_context(
        tracker_context=tracker_context,
        max_conversation_turns=COPILOT_ASSISTANT_TRACKER_MAX_TURNS,
        session_id=req.session_id,
    )

    return tracker_context


@langfuse.observe(capture_input=False, capture_output=False)
def get_relevant_assistant_files_for_copilot(
    project_generator: ProjectGenerator,
) -> BotFiles:
    """Get the relevant assistant files for the copilot.

    Args:
        project_generator: The project generator.

    Returns:
        The relevant assistant files.
    """
    # Copilot doesn't need to know about the docs and any file that is not a core
    # assistant file
    files = project_generator.get_bot_files(
        exclude_docs_directory=True,
        allowed_file_extensions=["yaml", "yml", "py", "jinja", "jinja2"],
    )

    # Track the retrieved assistant files
    CopilotLangfuseTelemetry.trace_copilot_relevant_assistant_files(
        relevant_assistant_files=files,
    )
    return files
