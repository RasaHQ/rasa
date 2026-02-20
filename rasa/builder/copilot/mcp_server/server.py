"""Main MCP server implementation for Rasa Copilot.

This server exposes Rasa-specific tools, resources, and prompts for external
MCP clients to use via SSE/HTTP.

It follows the FastMCP pattern from the official MCP documentation:
https://modelcontextprotocol.io/docs/develop/build-server
https://gofastmcp.com/servers/tools

This server runs as a webserver alongside the Sanic server, with the project
folder passed via the project_folder parameter to run_server().
"""

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from typing import Annotated, AsyncIterator, Optional

import structlog
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from rasa.builder.copilot.constants import RASA_PROJECT_FOLDER_ENV_VAR
from rasa.builder.copilot.mcp_server.constants import (
    INSTRUCTIONS_FILE_PATH,
    MCP_DEFAULT_HOST,
    MCP_DEFAULT_PORT,
    MCP_TOOL_GET_ASSISTANT_LOGS,
    MCP_TOOL_GET_DOMAIN_SCHEMA,
    MCP_TOOL_GET_E2E_SCHEMA,
    MCP_TOOL_GET_FLOW,
    MCP_TOOL_GET_FLOW_SCHEMA,
    MCP_TOOL_GET_RESPONSE,
    MCP_TOOL_GET_SLOT,
    MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS,
    MCP_TOOL_LIST_DEFAULT_ACTIONS,
    MCP_TOOL_LIST_DOMAIN_ACTIONS,
    MCP_TOOL_LIST_FLOWS,
    MCP_TOOL_LIST_RESPONSES,
    MCP_TOOL_LIST_SLOTS,
    MCP_TOOL_SEARCH_RASA_DOCS,
    MCP_TOOL_TALK_TO_ASSISTANT,
    MCP_TOOL_TRAIN_RASA_ASSISTANT,
    MCP_TOOL_VALIDATE_PROJECT,
    MCP_TRANSPORT_STDIO,
    MCP_TRANSPORT_STREAMABLE_HTTP,
)
from rasa.builder.copilot.mcp_server.models import (
    CustomActionsResponse,
    DocumentSearchResponse,
    GetFlowResponse,
    GetResponseResponse,
    GetSlotResponse,
    ListCustomActionsResponse,
    ListDefaultActionsResponse,
    ListFlowsResponse,
    ListResponsesResponse,
    ListSlotsResponse,
    SchemaResponse,
    SchemaType,
    TalkToAssistantResponse,
    TrainingResponse,
    ValidationResponse,
)
from rasa.shared.exceptions import RasaException

# NO heavy imports at module level - keep startup fast!
# All Rasa imports are lazy-loaded inside tools/resources/prompts

structlogger = structlog.get_logger()

# Project folder path, resolved once at startup by run_server().
# All MCP tools read this via _get_project_folder().
_project_folder_path: Optional[str] = None


@asynccontextmanager
async def dummy_progress_reporter(
    ctx: Context, interval_seconds: int = 10
) -> AsyncIterator[None]:
    """Context manager that reports progress periodically to keep MCP alive.

    Use this for long-running operations that don't have built-in progress
    reporting. It continuously increments a progress counter at regular
    intervals until the context exits.

    Args:
        ctx: The MCP context to report progress to
        interval_seconds: How often to report progress (default: 10 seconds)

    Example:
        async with dummy_progress_reporter(ctx):
            await some_long_running_operation()
    """
    cancelled = asyncio.Event()

    async def _report_progress() -> None:
        progress = 0
        while True:
            await asyncio.sleep(interval_seconds)
            if cancelled.is_set():
                return
            progress += 1
            try:
                await ctx.report_progress(progress=progress)
            except Exception:
                pass

    task = asyncio.create_task(_report_progress())
    try:
        yield
    finally:
        cancelled.set()
        task.cancel()
        # Suppress CancelledError from our deliberately cancelled task.
        # External cancellations will still propagate after this finally block.
        with suppress(asyncio.CancelledError):
            await task


def _load_instructions() -> str:
    """Load MCP server instructions from markdown file."""
    return INSTRUCTIONS_FILE_PATH.read_text()


# Initialize FastMCP server with metadata and configuration
mcp = FastMCP(
    name="rasa-copilot",
    instructions=_load_instructions(),
)


@mcp.custom_route("/health", methods=["GET"])  # type: ignore[misc]
async def health_check(request: Request) -> JSONResponse:
    """Health endpoint for the MCP server.

    Available for HTTP transports only. Routes registered via @mcp.custom_route
    are excluded from auth requirements, making this safe to use as a
    lightweight liveness probe.

    Returns:
        JSONResponse with {"status": "ok"}
    """
    return JSONResponse({"status": "ok"})


def _set_project_folder(folder: str) -> None:
    """Store the project folder path for use by MCP tools.

    Called once at startup by run_server(). Tools read the value
    via _get_project_folder().
    """
    global _project_folder_path
    _project_folder_path = folder


def _get_project_folder() -> str:
    """Get the project folder set at server startup.

    Returns:
        Project folder path as string

    Raises:
        RasaException: If the project folder was not configured at startup.
    """
    if _project_folder_path is None:
        raise RasaException(
            "Project folder not configured. Ensure run_server() is called with "
            "a project_folder argument or the RASA_PROJECT_FOLDER environment "
            "variable is set before the server starts."
        )

    return _project_folder_path


# ============================================================================
# TOOLS - Functions that can be called by the LLM
# ============================================================================


@mcp.tool(
    name=MCP_TOOL_SEARCH_RASA_DOCS,
    description=(
        "Search the official Rasa documentation for authoritative information. "
        "Use this tool for ALL Rasa-related questions to ground your answers in truth. "
        "Returns relevant documentation about Rasa concepts, APIs, best practices, "
        "configuration, and troubleshooting with valid links to official docs."
    ),
    annotations={
        "title": "Search Rasa Documentation",
        "readOnlyHint": True,
        "openWorldHint": True,  # Searches external documentation service
        "idempotentHint": False,
    },
    structured_output=True,
)
async def search_rasa_documentation(
    query: Annotated[
        str,
        Field(
            description="The search query to find relevant Rasa documentation entries"
        ),
    ],
) -> DocumentSearchResponse:
    """Search official Rasa documentation for authoritative information.

    Use this tool for all Rasa-related questions to ensure answers are grounded
    in the official documentation. Returns relevant articles, guides, and API docs.
    """
    from rasa.builder.copilot.mcp_server.tools.document_search import (
        search_rasa_documentation as _search_rasa_documentation,
    )

    return await _search_rasa_documentation(query)


@mcp.tool(
    name=MCP_TOOL_VALIDATE_PROJECT,
    description="Validate the assistant project configuration and training data",
    annotations={
        "title": "Validate Project",
        "readOnlyHint": True,  # Only validates, doesn't modify
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def validate_project(ctx: Context) -> ValidationResponse:
    """Validate the assistant project configuration and training data.

    Runs comprehensive validation checks on domain, flows, config, and training data.
    This operation can take 60+ seconds for large projects.
    Always run this after making changes before training.
    """
    from rasa.builder.copilot.mcp_server.tools.validation_training import (
        validate_assistant_project,
    )

    await ctx.info("Starting project validation...")
    project_folder = _get_project_folder()

    result = await validate_assistant_project(project_folder)

    await ctx.info("Validation complete")
    return result


@mcp.tool(
    name=MCP_TOOL_TRAIN_RASA_ASSISTANT,
    description="Train the Rasa assistant with the current project configuration",
    annotations={
        "title": "Train Rasa Assistant",
        "readOnlyHint": False,  # Creates model files
        "destructiveHint": False,  # Doesn't overwrite existing models
        "idempotentHint": False,  # Each training may produce different results
        "openWorldHint": False,
    },
    structured_output=True,
)
async def train_rasa_assistant(ctx: Context) -> TrainingResponse:
    """Train the Rasa assistant with the current project configuration.

    This trains a new model using the current domain, flows, and training data.
    Training can take several minutes for large projects.
    Only call this after validation passes successfully.
    """
    from rasa.builder.copilot.mcp_server.tools.validation_training import (
        train_assistant,
    )

    await ctx.info("Starting model training...")
    project_folder = _get_project_folder()

    async with dummy_progress_reporter(ctx):
        result = await train_assistant(project_folder)

    await ctx.info("Training complete")
    return result


@mcp.tool(
    name=MCP_TOOL_GET_ASSISTANT_LOGS,
    description="Get recent log entries from the Rasa assistant for troubleshooting",
    annotations={
        "title": "Assistant Logs",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": False,
    },
)
async def get_assistant_logs() -> str:
    """Get recent assistant logs for debugging and context.

    Provides access to the most recent log entries from the Rasa assistant,
    useful for troubleshooting errors and understanding system behavior.
    """
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        get_assistant_logs as _get_assistant_logs,
    )

    return await _get_assistant_logs()


@mcp.tool(
    name=MCP_TOOL_TALK_TO_ASSISTANT,
    description=(
        "Test the assistant by sending a sequence of messages and verifying responses. "
        "Use this to validate conversation flows work as expected after making changes."
    ),
    annotations={
        "title": "Talk to Assistant",
        "readOnlyHint": False,  # Creates conversation state
        "openWorldHint": False,
        "idempotentHint": False,  # Each call creates a new conversation
    },
    structured_output=True,
)
async def talk_to_assistant(
    ctx: Context,
    messages: Annotated[
        list[str],
        Field(
            description=(
                "List of user messages to send to the assistant in sequence. "
                "Each message will be sent one after another, waiting for the "
                "assistant's response before sending the next."
            ),
            examples=[
                ["Hello", "I want to book a flight", "To New York"],
                ["Hi there", "What can you help me with?"],
            ],
        ),
    ],
) -> TalkToAssistantResponse:
    """Test the assistant by sending messages and getting the conversation results.

    Sends each message to the trained assistant in order and returns:
    - The assistant's responses to each message
    - The complete tracker context showing conversation state, slots, and flow status

    Use this after training to verify that:
    - The assistant responds correctly to user inputs
    - Flows are triggered as expected
    - Slots are filled properly
    - The conversation follows the intended path

    Returns a structured response with the conversation history and tracker context.
    """
    from rasa.builder.copilot.mcp_server.tools.bot_interaction import (
        talk_to_assistant as _talk_to_assistant,
    )

    if not messages:
        return TalkToAssistantResponse(
            success=False,
            session_id="",
            message_count=0,
            conversation=[],
            tracker_context=None,
            error="No messages provided",
        )

    await ctx.info(f"Starting conversation with {len(messages)} message(s)...")

    async with dummy_progress_reporter(ctx):
        result = await _talk_to_assistant(messages)

    await ctx.info("Conversation completed")
    return result


@mcp.tool(
    name=MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS,
    description=(
        "List all custom actions in the project. Returns action name, class name, "
        "and file path (relative to project root) for each action. "
        "Use file_path to read the implementation.\n\n"
        "The tool automatically detects the actions folder from endpoints.yml or uses "
        "'actions' as default.\n\nResponse interpretation:\n"
        "- If error is set: Actions folder not found. "
        "Read endpoints.yml or ask user for location.\n"
        "- If count=0 and no error: Actions folder exists, "
        "but contains no actions (valid state).\n"
        "- If count>0: Successfully found actions."
    ),
    annotations={
        "title": "List Custom Actions",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_custom_actions(
    actions_folder: Annotated[
        Optional[str],
        Field(
            description=(
                "Path to the actions folder/package to scan, relative to "
                "project root. Can be a simple folder name (e.g., 'actions') "
                "or a nested path (e.g., 'my_package/actions'). If not "
                "provided, auto-detects from endpoints.yml or defaults to "
                "'actions'. Absolute paths and path traversal (..) are not "
                "allowed."
            )
        ),
    ] = None,
) -> CustomActionsResponse:
    """List all custom action implementations in the project.

    Scans Python files in the actions folder and returns information about
    each custom action class found, including:
    - Action name (from the name() method)
    - Class name
    - File path

    If actions_folder is not specified, attempts to detect it from
    endpoints.yml. Falls back to "actions" if not found.

    If the actions folder cannot be found, the error field will be set with guidance.
    In this case, read endpoints.yml or ask the user for the correct location.
    """
    from rasa.builder.copilot.mcp_server.tools.custom_actions import (
        list_custom_action_implementations,
    )

    project_folder = _get_project_folder()
    return await list_custom_action_implementations(project_folder, actions_folder)


@mcp.tool(
    name=MCP_TOOL_LIST_FLOWS,
    description=(
        "List all Rasa Flows definitions in the project. "
        "Returns flow ID, name, and file path for each flow. "
        "By default, searches in the 'data/' folder. "
        "Use data_folder to specify a different directory containing flow YAML files."
    ),
    annotations={
        "title": "List Project Flow Definitions",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_project_flow_definitions(
    data_folder: Optional[str] = "data",
) -> ListFlowsResponse:
    """List all flow definitions in the project.

    Args:
        data_folder: Folder containing flow files. Defaults to 'data/'.

    Scans for flow definition YAML files and returns basic metadata.
    Use get_project_file to read full flow definitions.
    """
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        list_project_flows,
    )

    project_folder = _get_project_folder()
    return await list_project_flows(project_folder, data_folder)


@mcp.tool(
    name=MCP_TOOL_LIST_SLOTS,
    description=(
        "List all Rasa Slot definitions from the project domain file(s). "
        "Returns slot name, type, and file path for each slot. "
        "By default, searches in the 'domain/' folder or 'domain.yml'. "
        "Use domain_folder to specify a directory with domain YAML files."
    ),
    annotations={
        "title": "List Project Slot Definitions",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_project_slot_definitions(
    domain_folder: Optional[str] = "domain",
) -> ListSlotsResponse:
    """List all slot definitions in the project's domain.

    Args:
        domain_folder: Folder containing domain files. Defaults to 'domain/'.

    Scans domain YAML files and returns basic slot metadata.
    Use get_project_file to read full slot configurations.
    """
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        list_project_slots,
    )

    project_folder = _get_project_folder()
    return await list_project_slots(project_folder, domain_folder)


@mcp.tool(
    name=MCP_TOOL_LIST_RESPONSES,
    description=(
        "List all response (utterances) definitions from the project domain file(s). "
        "Returns response name and file path for each response. "
        "By default, searches in the 'domain/' folder or 'domain.yml'. "
        "Use domain_folder to specify a directory with domain YAML files."
    ),
    annotations={
        "title": "List Project Response Definitions",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_project_response_definitions(
    domain_folder: Optional[str] = "domain",
) -> ListResponsesResponse:
    """List all response definitions in the project's domain.

    Args:
        domain_folder: Folder containing domain files. Defaults to 'domain/'.

    Scans domain YAML files and returns basic response metadata.
    Use get_project_file to read full response templates.
    """
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        list_project_responses,
    )

    project_folder = _get_project_folder()
    return await list_project_responses(project_folder, domain_folder)


@mcp.tool(
    name=MCP_TOOL_GET_FLOW,
    description=(
        "Get a single flow by flow ID (YAML key) or flow name. "
        "Returns flow metadata and full definition (steps, triggers, etc.)."
    ),
    annotations={
        "title": "Get Flow",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_flow(
    flow_id: Annotated[
        str,
        Field(description="Flow ID (YAML key) or human-readable flow name"),
    ],
    data_folder: Optional[str] = "data",
) -> GetFlowResponse:
    """Get a single flow by ID or name."""
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        get_project_flow,
    )

    project_folder = _get_project_folder()
    return await get_project_flow(project_folder, flow_id, data_folder)


@mcp.tool(
    name=MCP_TOOL_GET_SLOT,
    description=(
        "Get a single slot by name. "
        "Returns slot metadata and full definition from the domain."
    ),
    annotations={
        "title": "Get Slot",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_slot(
    slot_name: Annotated[str, Field(description="Slot name")],
    domain_folder: Optional[str] = "domain",
) -> GetSlotResponse:
    """Get a single slot by name."""
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        get_project_slot,
    )

    project_folder = _get_project_folder()
    return await get_project_slot(project_folder, slot_name, domain_folder)


@mcp.tool(
    name=MCP_TOOL_GET_RESPONSE,
    description=(
        "Get a single response (utterance) by name. "
        "Returns response metadata and full definition (text, image, buttons, etc.)."
    ),
    annotations={
        "title": "Get Response",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_response(
    response_name: Annotated[
        str, Field(description="Response name (e.g. utter_greet)")
    ],
    domain_folder: Optional[str] = "domain",
) -> GetResponseResponse:
    """Get a single response by name."""
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        get_project_response,
    )

    project_folder = _get_project_folder()
    return await get_project_response(project_folder, response_name, domain_folder)


@mcp.tool(
    name=MCP_TOOL_LIST_DOMAIN_ACTIONS,
    description=(
        "List all Rasa's Custom Actions that are declared in the domain file(s). "
        "Returns action name and file path where the action is registered. "
        "By default, searches in the 'domain/' folder or 'domain.yml'. "
        "Use domain_folder to specify a directory with domain YAML files. "
        "Note: This lists domain declarations, not Python implementations."
    ),
    annotations={
        "title": "List Project Custom Actions in Domain",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_project_custom_actions_in_domain(
    domain_folder: Optional[str] = "domain",
) -> ListCustomActionsResponse:
    """List all custom action declarations in the project's domain.

    Args:
        domain_folder: Folder containing domain files. Defaults to 'domain/'.

    Scans domain YAML files for action declarations.
    This returns domain config, not Python implementations.
    """
    from rasa.builder.copilot.mcp_server.tools.project_context import (
        list_project_custom_actions,
    )

    project_folder = _get_project_folder()
    return await list_project_custom_actions(project_folder, domain_folder)


@mcp.tool(
    name=MCP_TOOL_LIST_DEFAULT_ACTIONS,
    description=(
        "List all default/built-in action names provided by Rasa. "
        "These actions are available without any configuration. "
        "Users can override these to customize behavior. "
        "See Rasa documentation for details on what each action does."
    ),
    annotations={
        "title": "List Default Action Names",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_default_action_names() -> ListDefaultActionsResponse:
    """List all default/built-in action names provided by Rasa.

    Returns the list of action names that Rasa provides out of the box.
    Users can override these in their project to customize behavior.
    See Rasa documentation for details on each action's purpose.
    """
    from rasa.shared.core.constants import DEFAULT_ACTION_NAMES

    return ListDefaultActionsResponse(
        success=True,
        actions=list(DEFAULT_ACTION_NAMES),
    )


@mcp.tool(
    name=MCP_TOOL_GET_FLOW_SCHEMA,
    description=(
        "Get the official Rasa flow schema. Returns a JSON Schema (the schema "
        "document itself is in JSON format). Use to validate assistant flows in "
        "project YAML files or to generate new flows. The schema describes: flow "
        "name and description, step types, branching logic (if/then/else), "
        "calling another flow, collect steps and slots, flow guards, and more. "
        "By default returns the CALM-only flow schema without NLU-related "
        "properties."
    ),
    annotations={
        "title": "Get Flow Schema",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_flow_schema(calm_only: bool = True) -> SchemaResponse:
    """Return the Rasa flow schema (JSON Schema format)."""
    from rasa.builder.copilot.mcp_server.tools.rasa_schemas import (
        get_flow_schema as _get_flow_schema,
    )

    try:
        return _get_flow_schema(calm_only=calm_only)
    except Exception as e:
        return SchemaResponse(
            success=False,
            schema_type=SchemaType.FLOW,
            schema_content="",
            error=str(e),
        )


@mcp.tool(
    name=MCP_TOOL_GET_DOMAIN_SCHEMA,
    description=(
        "Get the official Rasa domain schema. Returns a YAML schema (the schema "
        "document is in YAML schema format). Use to validate domain YAML or generate "
        "domain files. The schema describes: slots, custom actions, responses "
        "(utterance templates), and more. By default returns the CALM-only domain "
        "schema without NLU-related properties."
    ),
    annotations={
        "title": "Get Domain Schema",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_domain_schema(calm_only: bool = True) -> SchemaResponse:
    """Return the Rasa domain schema in YAML schema format (includes responses).

    Args:
        calm_only: Whether to return only the CALM-related properties of the domain
            schema.
    """
    from rasa.builder.copilot.mcp_server.tools.rasa_schemas import (
        get_full_domain_schema as _get_full_domain_schema,
    )

    try:
        return _get_full_domain_schema(calm_only=calm_only)

    except Exception as e:
        return SchemaResponse(
            success=False,
            schema_type=SchemaType.DOMAIN,
            schema_content="",
            error=str(e),
        )


@mcp.tool(
    name=MCP_TOOL_GET_E2E_SCHEMA,
    description=(
        "Get the official Rasa Assistant E2E test schema. Returns a YAML schema (the "
        "schema document is in YAML schema format). Use to validate e2e test YAML or "
        "generate e2e tests. The schema describes: test cases, steps (user and bot "
        "messages), fixtures, metadata, stub custom actions, and assertions."
    ),
    annotations={
        "title": "Get E2E Schema",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_e2e_schema() -> SchemaResponse:
    """Return the Rasa e2e test schema in YAML schema format."""
    from rasa.builder.copilot.mcp_server.tools.rasa_schemas import (
        get_e2e_schema as _get_e2e_schema,
    )

    try:
        return _get_e2e_schema()
    except Exception as e:
        return SchemaResponse(
            success=False,
            schema_type=SchemaType.E2E,
            schema_content="",
            error=str(e),
        )


# ============================================================================
# RESOURCES - File-like static data that can be read by clients
# ============================================================================

# Note: Dynamic resource templates (uri_template) are not
# supported in mcp.server.fastmcp


# ============================================================================
# PROMPTS - Pre-written templates that help accomplish specific tasks
# ============================================================================


@mcp.prompt(
    name="system_prompt",
    title="Rasa Copilot System Prompt",
    description=(
        "Main system prompt for the Rasa copilot assistant with comprehensive "
        "instructions and guidelines"
    ),
)
async def system_prompt() -> list:
    """Main system prompt for the Rasa copilot assistant.

    Provides the complete system prompt including tool usage patterns,
    conversation guidelines, and best practices for building Rasa assistants.
    """
    from rasa.builder.copilot.mcp_server.prompts.prompt_loader import (
        get_copilot_system_prompt,
    )

    content = await get_copilot_system_prompt()
    return [{"role": "user", "content": {"type": "text", "text": content}}]


@mcp.prompt(
    name="user_message_context",
    title="User Message Context",
    description=(
        "Template for enriching user messages with additional context about the "
        "current session"
    ),
)
async def user_message_context() -> list:
    """Template for adding context to user messages.

    Provides a prompt template that helps contextualize user messages
    with information about the current project state and conversation history.
    """
    from rasa.builder.copilot.mcp_server.prompts.prompt_loader import (
        get_last_user_message_context_prompt,
    )

    content = await get_last_user_message_context_prompt()
    return [{"role": "user", "content": {"type": "text", "text": content}}]


@mcp.prompt(
    name="training_error_analysis",
    title="Training Error Analysis",
    description="Template for analyzing and fixing training errors in Rasa projects",
)
async def training_error_analysis() -> list:
    """Template for analyzing training errors.

    Provides a structured approach for analyzing validation and training errors,
    helping to identify root causes and suggest fixes.
    """
    from rasa.builder.copilot.mcp_server.prompts.prompt_loader import (
        get_training_error_handler_prompt,
    )

    content = await get_training_error_handler_prompt()
    return [{"role": "user", "content": {"type": "text", "text": content}}]


# ============================================================================
# SERVER INITIALIZATION
# ============================================================================


def run_server(
    host: str = MCP_DEFAULT_HOST,
    port: int = MCP_DEFAULT_PORT,
    transport: str = MCP_TRANSPORT_STREAMABLE_HTTP,
    project_folder: Optional[str] = None,
) -> None:
    """Run the MCP server.

    This is the entry point for the MCP server. It can be started standalone
    via `rasa tools run` or embedded alongside the Sanic server (builder mode).

    The resolved project folder is stored as module-level state so that MCP
    tools can access it via _get_project_folder() without reading environment
    variables at request time.

    IMPORTANT: When using stdio transport, the caller MUST ensure all logging
    goes to stderr, not stdout. The MCP stdio protocol requires stdout to
    contain ONLY JSON-RPC messages. Any other output will corrupt the protocol
    and break IDE client connections.

    Args:
        host: Host to bind the server to (only used for streamable-http).
        port: Port to bind the server to (only used for streamable-http).
        transport: FastMCP transport to use – MCP_TRANSPORT_STDIO or
            MCP_TRANSPORT_STREAMABLE_HTTP (default: MCP_TRANSPORT_STREAMABLE_HTTP
            for backward compatibility with the embedded builder server).
        project_folder: Path to the Rasa project folder. If not provided,
            falls back to the RASA_PROJECT_FOLDER environment variable.
    """
    try:
        resolved_folder = project_folder or os.getenv(RASA_PROJECT_FOLDER_ENV_VAR)
        if resolved_folder:
            _set_project_folder(resolved_folder)

        structlogger.info(
            "mcp_server.server.starting",
            event_info="Starting MCP server",
            transport=transport,
            project_folder=resolved_folder,
            host=host if transport != MCP_TRANSPORT_STDIO else None,
            port=port if transport != MCP_TRANSPORT_STDIO else None,
        )

        if transport == MCP_TRANSPORT_STDIO:
            structlogger.info(
                "mcp_server.server.ready",
                event_info="MCP server ready (stdio mode)",
                transport=transport,
            )
            mcp.run(transport=MCP_TRANSPORT_STDIO)
        else:
            mcp.settings.host = host
            mcp.settings.port = port
            structlogger.info(
                "mcp_server.server.ready",
                event_info="MCP server ready (http mode)",
                transport=transport,
                host=host,
                port=port,
                url=f"http://{host}:{port}/mcp",
                health_url=f"http://{host}:{port}/health",
            )
            mcp.run(transport=MCP_TRANSPORT_STREAMABLE_HTTP)

    except Exception as e:
        structlogger.error(
            "mcp_server.server.startup_error",
            event_info="Failed to start MCP server",
            error=str(e),
        )
        import sys

        sys.exit(1)
