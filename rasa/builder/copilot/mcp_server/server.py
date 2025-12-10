"""Main MCP server implementation for Rasa Copilot.

This server exposes tools, resources, and prompts for the Agents SDK to use
via SSE/HTTP.
It follows the FastMCP pattern from the official MCP documentation:
https://modelcontextprotocol.io/docs/develop/build-server
https://gofastmcp.com/servers/tools

This server runs as a webserver alongside the Sanic server, with the project
folder passed via RASA_PROJECT_FOLDER environment variable.
"""

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from typing import Annotated, AsyncIterator, Dict

import structlog
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from rasa.builder.copilot.mcp_server.models import (
    DocumentSearchResponse,
    FileContentResponse,
    FileListResponse,
    MultiFileUpdate,
    ReadFilesResponse,
    TalkToAssistantResponse,
    TrainingResponse,
    UpdateFilesResponse,
    ValidationResponse,
    WriteFileResponse,
)
from rasa.shared.exceptions import RasaException

# NO heavy imports at module level - keep startup fast!
# All Rasa imports are lazy-loaded inside tools/resources/prompts

structlogger = structlog.get_logger()


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


# Initialize FastMCP server with metadata and configuration
mcp = FastMCP(
    name="rasa-copilot",
    instructions=(
        "MCP server for Rasa assistant development with "
        "tools for file operations, validation, and training"
    ),
)


def _get_project_folder() -> str:
    """Get the project folder from environment.

    The project folder is passed via RASA_PROJECT_FOLDER environment variable
    when the subprocess is spawned.

    Returns:
        Project folder path as string
    """
    project_folder = os.getenv("RASA_PROJECT_FOLDER")

    if project_folder is None:
        raise RasaException(
            "Project folder not configured. The RASA_PROJECT_FOLDER environment "
            "variable must be set by the parent copilot process."
        )

    return project_folder


# ============================================================================
# TOOLS - Functions that can be called by the LLM
# ============================================================================


@mcp.tool(
    description=(
        "Search Rasa documentation for relevant information about concepts, "
        "APIs, and best practices"
    ),
    annotations={
        "title": "Search Documentation",
        "readOnlyHint": True,
        "openWorldHint": True,  # Searches external documentation service
        "idempotentHint": True,
    },
    structured_output=True,
)
async def search_docs(
    query: Annotated[
        str,
        Field(
            description="The search query to find relevant Rasa documentation entries"
        ),
    ],
) -> DocumentSearchResponse:
    """Search Rasa documentation for relevant information.

    This tool queries the Rasa documentation service to find relevant articles,
    guides, and API documentation based on your search terms.
    """
    from rasa.builder.copilot.mcp_server.tools.document_search import (
        search_rasa_documentation,
    )

    return await search_rasa_documentation(query)


@mcp.tool(
    description="Read all bot project files with their complete contents",
    annotations={
        "title": "Read All Project Files",
        "readOnlyHint": True,
        "openWorldHint": False,  # Local filesystem only
        "idempotentHint": True,
    },
    structured_output=True,
)
async def read_project_files(
    exclude_docs: Annotated[
        bool, Field(description="Whether to exclude the docs directory from results")
    ] = True,
    allowed_extensions: Annotated[
        str,
        Field(
            description=(
                "Comma-separated list of file extensions to "
                "include (e.g., 'yaml,yml,py')"
            )
        ),
    ] = "yaml,yml,py,jinja,jinja2",
) -> ReadFilesResponse:
    """Read all bot project files with their complete contents.

    Returns all files in the project that match the allowed extensions,
    organized as a mapping of file paths to their contents.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import (
        read_assistant_files,
    )

    project_folder = _get_project_folder()
    return await read_assistant_files(project_folder, exclude_docs, allowed_extensions)


@mcp.tool(
    description="Get the content of a specific file in the bot project",
    annotations={
        "title": "Read Single File",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def get_project_file(
    file_path: Annotated[
        str,
        Field(
            description=(
                "Relative path to the file "
                "(e.g., 'domain.yml', 'flows/greeting.yml')"
            )
        ),
    ],
) -> FileContentResponse:
    """Get the content of a specific file in the bot project.

    Use this to read individual files when you know the exact path.
    More efficient than reading all files when you only need one.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import get_file_content

    project_folder = _get_project_folder()
    return await get_file_content(project_folder, file_path)


@mcp.tool(
    description="List all files in the bot project as a directory tree",
    annotations={
        "title": "List Project Files",
        "readOnlyHint": True,
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def list_project_files() -> FileListResponse:
    """List all files in the bot project as a directory tree.

    Shows the complete directory structure with all files organized by folder.
    Use this first to understand the project layout before reading specific files.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import list_files

    project_folder = _get_project_folder()
    return await list_files(project_folder)


@mcp.tool(
    description="Write or overwrite a file in the bot project",
    annotations={
        "title": "Write Project File",
        "readOnlyHint": False,
        "destructiveHint": True,  # Overwrites existing files
        "idempotentHint": True,  # Same content = same result
        "openWorldHint": False,
    },
    structured_output=True,
)
async def write_project_file(
    file_path: Annotated[
        str,
        Field(
            description=(
                "Relative path to the file "
                "(e.g., 'domain/slots.yml', 'flows/greeting.yml')"
            )
        ),
    ],
    content: Annotated[str, Field(description="Complete content to write to the file")],
) -> WriteFileResponse:
    """Write or overwrite a file in the bot project.

    Creates new files or completely replaces existing file contents.
    Always validate the project after making changes.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import write_file

    project_folder = _get_project_folder()
    return await write_file(project_folder, file_path, content)


@mcp.tool(
    description="Write multiple files in a single coordinated operation",
    annotations={
        "title": "Update Multiple Files",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    structured_output=True,
)
async def update_multiple_files(
    files: Annotated[
        Dict[str, str],
        Field(
            description="Dictionary mapping file paths to their new contents",
            examples=[{"domain/slots.yml": "slots:\n  user_name:\n    type: text"}],
        ),
    ],
) -> UpdateFilesResponse:
    """Write multiple files in a single coordinated operation.

    Use this for making related changes across multiple files atomically.
    More efficient than multiple individual write operations.
    Automatically validates all file paths for security.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import update_files

    # Validate input using Pydantic model
    validated = MultiFileUpdate(files=files)

    project_folder = _get_project_folder()
    return await update_files(project_folder, validated.files)


@mcp.tool(
    description="Validate the bot project configuration and training data",
    annotations={
        "title": "Validate Project",
        "readOnlyHint": True,  # Only validates, doesn't modify
        "openWorldHint": False,
        "idempotentHint": True,
    },
    structured_output=True,
)
async def validate_project(ctx: Context) -> ValidationResponse:
    """Validate the bot project configuration and training data.

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
    description="Train a new bot model with the current project configuration",
    annotations={
        "title": "Train Model",
        "readOnlyHint": False,  # Creates model files
        "destructiveHint": False,  # Doesn't overwrite existing models
        "idempotentHint": False,  # Each training may produce different results
        "openWorldHint": False,
    },
    structured_output=True,
)
async def train_model(ctx: Context) -> TrainingResponse:
    """Train a new bot model with the current project configuration.

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
    from rasa.builder.copilot.mcp_server.resources.project_context import (
        get_assistant_logs,
    )

    return await get_assistant_logs()


@mcp.tool(
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
                "bot's response before sending the next."
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
    - The bot's responses to each message
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


# ============================================================================
# RESOURCES - File-like data that can be read by clients
# ============================================================================


@mcp.resource(
    uri="project://files",
    name="project_files",
    title="Complete Project Files",
    description=(
        "All project files with their complete contents (domain, flows, "
        "config, actions, training data). Use this FIRST to avoid "
        "multiple tool calls."
    ),
    mime_type="application/json",
)
async def project_files_resource() -> str:
    """Get all project files with complete contents in one resource.

    This resource provides the entire project structure and file contents upfront,
    eliminating the need for multiple get_project_file() tool calls. This is the
    most efficient way to explore the project and saves agent turns.

    Includes: domain files, flows, config, actions, prompts, and training data.
    Excludes: docs folder, hidden files, models, __pycache__.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import (
        read_assistant_files,
    )

    project_folder = _get_project_folder()
    result = await read_assistant_files(
        project_folder,
        exclude_docs=True,
        allowed_extensions="yaml,yml,py,jinja,jinja2,md",
    )
    return result.model_dump_json()


@mcp.resource(
    uri="project://tree",
    name="project_tree",
    title="Project File Tree",
    description=(
        "Lightweight file tree structure showing all project files and directories "
        "without content"
    ),
    mime_type="application/json",
)
async def project_tree_resource() -> str:
    """Get project file tree structure without file contents.

    This lightweight resource shows the complete directory structure and file list,
    useful for understanding project organization without loading all file contents.
    Use this when you only need to see what files exist.
    """
    from rasa.builder.copilot.mcp_server.tools.file_operations import list_files

    project_folder = _get_project_folder()
    result = await list_files(project_folder)
    return result.model_dump_json()


@mcp.resource(
    uri="project://flows",
    name="project_flows",
    title="All Flow Definitions",
    description=(
        "All conversation flow files (data/*.yml) showing assistant skills and "
        "conversation patterns"
    ),
    mime_type="application/json",
)
async def project_flows_resource() -> str:
    """Get all flow definition files.

    Returns all YAML files from the data/ directory containing conversation flows.
    Each flow defines a conversational skill or pattern the assistant can handle.
    """
    from rasa.builder.copilot.mcp_server.resources.project_context import (
        get_flows_definitions,
    )

    project_folder = _get_project_folder()
    return await get_flows_definitions(project_folder)


@mcp.resource(
    uri="project://domain",
    name="project_domain",
    title="All Domain Definitions",
    description=(
        "All domain files (slots, responses, actions) defining the assistant's "
        "memory and capabilities"
    ),
    mime_type="application/json",
)
async def project_domain_resource() -> str:
    """Get all domain definition files.

    Returns all YAML files from the domain/ directory containing:
    - Slots (assistant memory)
    - Responses (what the assistant can say)
    - Actions (custom logic the assistant can run)
    """
    from rasa.builder.copilot.mcp_server.resources.project_context import (
        get_domain_definitions,
    )

    project_folder = _get_project_folder()
    return await get_domain_definitions(project_folder)


@mcp.resource(
    uri="project://actions",
    name="project_actions",
    title="Custom Action Code",
    description="All Python custom action implementations from actions/ directory",
    mime_type="application/json",
)
async def project_actions_resource() -> str:
    """Get all custom action Python files.

    Returns all Python files from the actions/ directory containing
    custom action implementations for complex logic.
    """
    from rasa.builder.copilot.mcp_server.resources.project_context import (
        get_custom_actions_code,
    )

    project_folder = _get_project_folder()
    return await get_custom_actions_code(project_folder)


# Note: Dynamic resource templates (uri_template) are not
# supported in mcp.server.fastmcp
# The get_project_file tool provides the same functionality for
# accessing individual files


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


def run_server(host: str = "127.0.0.1", port: int = 5051) -> None:
    """Run the MCP server with SSE transport.

    This is the entry point for the MCP webserver. It's started alongside the
    Sanic server with RASA_PROJECT_FOLDER set in the environment.

    The server uses SSE (Server-Sent Events) over HTTP for communication.

    Args:
        host: Host to bind the server to (default: 127.0.0.1)
        port: Port to bind the server to (default: 5051)
    """
    try:
        project_folder = os.getenv("RASA_PROJECT_FOLDER")

        structlogger.info(
            "mcp_server.server.starting",
            event_info="Starting MCP streamable-http server",
            project_folder=project_folder,
            host=host,
            port=port,
        )

        # FastMCP with SSE transport
        # This serves the MCP server over HTTP with SSE for streaming
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")

    except Exception as e:
        structlogger.error(
            "mcp_server.server.startup_error",
            event_info="Failed to start MCP server",
            error=str(e),
        )
        import sys

        sys.exit(1)
