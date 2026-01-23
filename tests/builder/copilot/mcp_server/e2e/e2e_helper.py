"""Helper utilities for MCP server end-to-end testing."""

import json
import os
import sys
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.types import CallToolResult, TextContent

# Debug mode: set DEBUG_MCP_RESPONSE=1 to see raw MCP responses
# Note: Use pytest -s flag to see debug output (pytest captures stdout by default)
DEBUG_MCP_RESPONSE = os.environ.get("DEBUG_MCP_RESPONSE", "").lower() in (
    "1",
    "true",
    "yes",
)


async def call_tool_safely(
    session: ClientSession, tool_name: str, arguments: Dict[str, Any]
) -> CallToolResult:
    """Call an MCP tool and return the result.

    Args:
        session: MCP client session
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        CallToolResult from the tool execution
    """
    result = await session.call_tool(tool_name, arguments)
    print_raw_response(result, tool_name)
    return result


def print_raw_response(result: CallToolResult, tool_name: str = "") -> None:
    """Print raw MCP response for debugging.

    Args:
        result: CallToolResult from tool execution
        tool_name: Optional tool name for context

    Note: Use pytest -s flag to see this output (pytest captures stdout by default)
    """
    if not DEBUG_MCP_RESPONSE:
        return

    # Use stderr so output is visible even if pytest captures stdout
    # (though -s flag is still recommended for full visibility)
    print("\n" + "=" * 80, file=sys.stderr)
    if tool_name:
        print(f"[DEBUG] Raw MCP Response for tool: {tool_name}", file=sys.stderr)
    else:
        print("[DEBUG] Raw MCP Response", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"isError: {result.isError}", file=sys.stderr)
    print(f"structuredContent: {result.structuredContent}", file=sys.stderr)
    print(f"content: {result.content}", file=sys.stderr)
    if result.content:
        print("\nContent details:", file=sys.stderr)
        for i, content_item in enumerate(result.content):
            print(f"  Content item {i}:", file=sys.stderr)
            print(f"    Type: {type(content_item)}", file=sys.stderr)
            if isinstance(content_item, TextContent):
                print(f"    Text: {content_item.text}", file=sys.stderr)
                print(f"    Type (content): {content_item.type}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)


def parse_tool_result(result: CallToolResult) -> Dict[str, Any]:
    """Parse tool result content into a dictionary.

    Args:
        result: CallToolResult from tool execution

    Returns:
        Parsed result as dictionary
    """
    if result.isError:
        return {"error": True, "message": str(result.content)}

    if result.structuredContent:
        return result.structuredContent

    if result.content:
        # Try to parse as JSON
        for content_item in result.content:
            if isinstance(content_item, TextContent):
                try:
                    return json.loads(content_item.text)
                except json.JSONDecodeError:
                    return {"text": content_item.text}

    return {}


async def get_resource_by_uri(session: ClientSession, uri: str) -> Optional[Any]:
    """Get a resource definition by URI.

    Args:
        session: MCP client session
        uri: Resource URI

    Returns:
        Resource definition or None if not found
    """
    resources = await session.list_resources()
    for resource in resources.resources:
        # Convert URI to string for comparison (resource.uri may be AnyUrl object)
        if str(resource.uri) == uri:
            return resource
    return None


def assert_file_exists(result: Dict[str, Any], file_path: str) -> None:
    """Assert that a file operation result indicates the file exists.

    Args:
        result: Parsed tool result
        file_path: Expected file path
    """
    error_msg = f"File {file_path} should exist"
    assert "error" not in result or not result["error"], error_msg
    if "exists" in result:
        assert result["exists"], f"File {file_path} should exist"


def assert_operation_success(result: Dict[str, Any]) -> None:
    """Assert that operation was successful.

    Args:
        result: Parsed operation result
    """
    assert result.get("success") is True, f"Operation should succeed: {result}"


def assert_training_success(result: Dict[str, Any]) -> None:
    """Assert that training was successful.

    Args:
        result: Parsed training result (should match TrainingResponse structure)
    """
    if not result.get("success", False):
        # Include the full error message in the assertion
        error_msg = result.get("message", "Unknown error")
        full_error = (
            f"Training should succeed but failed.\n\n"
            f"Error message: {error_msg}\n\n"
            f"Full result: {result}"
        )
        assert False, full_error

    # When success=True, model_path should always be present
    model_path = result.get("model_path")
    if model_path is None:
        full_error = (
            f"Training reported success but model_path is missing.\n\n"
            f"Message: {result.get('message', 'N/A')}\n"
            f"Full result: {result}"
        )
        assert False, full_error
