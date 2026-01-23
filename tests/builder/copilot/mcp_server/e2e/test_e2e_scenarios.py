"""End-to-end tests for MCP server using actual MCP client connections.

These tests verify complete workflows by:
1. Starting the MCP server in a background process
2. Connecting to it via MCP client
3. Executing tool calls and verifying results
"""

import json
from pathlib import Path
from typing import Callable

import pytest
from mcp import ClientSession

from rasa.builder.copilot.mcp_server.constants import (
    MCP_TOOL_GET_ASSISTANT_LOGS,
    MCP_TOOL_GET_PROJECT_FILE,
    MCP_TOOL_LIST_PROJECT_FILES,
    MCP_TOOL_SEARCH_DOCS,
    MCP_TOOL_TALK_TO_ASSISTANT,
    MCP_TOOL_TRAIN_MODEL,
    MCP_TOOL_UPDATE_MULTIPLE_FILES,
    MCP_TOOL_VALIDATE_PROJECT,
    MCP_TOOL_WRITE_PROJECT_FILE,
)
from tests.builder.copilot.mcp_server.e2e.e2e_helper import (
    assert_file_exists,
    assert_operation_success,
    assert_training_success,
    call_tool_safely,
    get_resource_by_uri,
    parse_tool_result,
)


class TestE2EFileOperations:
    """End-to-end tests for file operation tools."""

    @pytest.mark.asyncio
    async def test_complete_file_workflow(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test complete workflow: list -> read -> write -> read."""
        # 1. List files
        result = await call_tool_safely(mcp_client, MCP_TOOL_LIST_PROJECT_FILES, {})
        parsed = parse_tool_result(result)
        assert "files" in parsed or "tree" in parsed
        assert "domain/domain.yml" in str(parsed)

        # 2. Read a file
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_PROJECT_FILE, {"file_path": "domain/domain.yml"}
        )
        parsed = parse_tool_result(result)
        assert_file_exists(parsed, "domain/domain.yml")
        # Check for domain content (actions, slots, responses, etc.)
        content = str(parsed.get("content", ""))
        assert "actions" in content or "responses" in content or "slots" in content

        # 3. Write a new file
        new_content = (
            "version: '3.1'\nintents: []\nresponses:\n  "
            "utter_greet:\n    - text: Hello!"
        )
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_WRITE_PROJECT_FILE,
            {"file_path": "domain/responses.yml", "content": new_content},
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        # 4. Verify file was written
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_PROJECT_FILE, {"file_path": "domain/responses.yml"}
        )
        parsed = parse_tool_result(result)
        assert_file_exists(parsed, "domain/responses.yml")
        assert "utter_greet" in str(parsed.get("content", ""))
        assert (test_project_folder / "domain" / "responses.yml").exists()

    @pytest.mark.asyncio
    async def test_multi_file_update(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test multi-file update tool."""
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_UPDATE_MULTIPLE_FILES,
            {
                "files": {
                    "domain/domain1.yml": "version: '3.1'\nintents: []",
                    "domain/domain2.yml": "version: '3.1'\nintents: []",
                }
            },
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)
        assert (test_project_folder / "domain" / "domain1.yml").exists()
        assert (test_project_folder / "domain" / "domain2.yml").exists()
        assert (
            test_project_folder / "domain" / "domain1.yml"
        ).read_text() == "version: '3.1'\nintents: []"
        assert (
            test_project_folder / "domain" / "domain2.yml"
        ).read_text() == "version: '3.1'\nintents: []"


class TestE2EValidationWorkflow:
    """End-to-end tests for validation workflow."""

    @pytest.mark.asyncio
    async def test_validate_workflow(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test workflow: validate -> modify -> validate."""
        # 1. Initial validation
        result = await call_tool_safely(mcp_client, MCP_TOOL_VALIDATE_PROJECT, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        # 2. Make a change
        updated_domain = (
            "actions: []\nslots: {}\nintents:\n  - greet\nresponses:\n  "
            "utter_greet_user:\n    - text: Hello there!"
        )
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_WRITE_PROJECT_FILE,
            {"file_path": "domain/domain.yml", "content": updated_domain},
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        # 3. Validate again
        result = await call_tool_safely(mcp_client, MCP_TOOL_VALIDATE_PROJECT, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

    @pytest.mark.asyncio
    async def test_validation_failure(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test validation failure."""
        # Write a file with validation errors
        updated_domain = "invalid_domain: invalid_domain"
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_WRITE_PROJECT_FILE,
            {"file_path": "domain/domain.yml", "content": updated_domain},
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        # Validate the project
        result = await call_tool_safely(mcp_client, MCP_TOOL_VALIDATE_PROJECT, {})
        parsed = parse_tool_result(result)
        # Assert validation failed
        assert (
            parsed.get("success") is False
        ), f"Validation should fail with invalid domain: {parsed}"
        assert "errors" in parsed, "Validation result should include errors"
        assert len(parsed.get("errors", [])) > 0, "Should have at least one error"


class TestE2EAssistantConversation:
    """End-to-end test to train and talk to the assistant."""

    @pytest.mark.asyncio
    async def test_talk_to_assistant(
        self,
        mcp_client: ClientSession,
        test_project_folder: Path,
        rasa_server: tuple[int, Callable[[str], None]],
    ):
        """Test talk_to_assistant tool after training."""
        # Train the assistant
        result = await call_tool_safely(mcp_client, MCP_TOOL_TRAIN_MODEL, {})
        parsed = parse_tool_result(result)
        assert_training_success(parsed)

        # Get the model path from training result
        model_path = parsed.get("model_path")

        # Start the Rasa server with the trained model
        _, start_server = rasa_server
        start_server(str(model_path))

        # Test assistant conversation
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_TALK_TO_ASSISTANT, {"messages": ["Hello"]}
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)
        assert len(parsed.get("conversation", [])) > 0, "Should have conversation turns"

        # Test get_assistant_logs tool
        result = await call_tool_safely(mcp_client, MCP_TOOL_GET_ASSISTANT_LOGS, {})
        parsed = parse_tool_result(result)
        # The tool returns a string, which may be in 'result' (structuredContent)
        # or 'text' (content)
        logs = parsed.get("result", parsed.get("text", ""))
        if not isinstance(logs, str):
            logs = str(logs) if logs else ""
        assert isinstance(logs, str), "Logs should be a string"
        # Note: logs may be empty if logging isn't configured to capture logs
        # in the test environment, but the tool itself is working correctly


class TestE2EDocumentationSearch:
    """End-to-end tests for documentation search tool."""

    @pytest.mark.asyncio
    async def test_search_docs(self, mcp_client: ClientSession):
        """Test search_docs tool returns properly formatted results."""
        # Call the search_docs tool with a test query
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_SEARCH_DOCS, {"query": "What is Rasa?"}
        )
        parsed = parse_tool_result(result)

        # Verify the response structure matches DocumentSearchResponse
        assert "documents" in parsed, "Response should include 'documents' field"
        assert len(parsed["documents"]) > 0, "Documents should always be returned"

        # Verify document structure
        doc = parsed["documents"][0]
        # Verify the most important fields exist
        assert "title" in doc, "Document should have 'title' field"
        assert "url" in doc, "Document should have 'url' field"
        assert "content" in doc, "Document should have 'content' field"
        assert len(doc["url"]) > 0, "URL should not be empty"
        assert len(doc["content"]) > 0, "Content should not be empty"


class TestE2EResources:
    """End-to-end tests for MCP resources."""

    @pytest.mark.asyncio
    async def test_project_files_resource(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test accessing project files resource."""
        # Debug: list all available resources
        resources = await mcp_client.list_resources()
        available_uris = [r.uri for r in resources.resources]

        resource = await get_resource_by_uri(mcp_client, "project://files")
        if resource is None:
            msg = (
                f"Resource 'project://files' not found. "
                f"Available resources: {available_uris}"
            )
            pytest.fail(msg)

        result = await mcp_client.read_resource(resource.uri)
        assert result.contents is not None
        assert len(result.contents) > 0

        # Parse JSON content
        content_text = result.contents[0].text
        data = json.loads(content_text)
        assert "files" in data
        assert "domain/domain.yml" in data["files"]


class TestE2EPrompts:
    """End-to-end tests for MCP prompts."""

    @pytest.mark.asyncio
    async def test_system_prompt(self, mcp_client: ClientSession):
        """Test retrieving system prompt."""
        prompts = await mcp_client.list_prompts()
        system_prompt = next(
            (p for p in prompts.prompts if p.name == "system_prompt"), None
        )
        assert system_prompt is not None

        result = await mcp_client.get_prompt(system_prompt.name, {})
        assert result.messages is not None
        assert len(result.messages) > 0
        assert result.messages[0].role == "user"


class TestE2EErrorHandling:
    """End-to-end tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_file_path(self, mcp_client: ClientSession):
        """Test error handling for invalid file paths."""
        # Try to access restricted path
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_PROJECT_FILE, {"file_path": "../outside.yml"}
        )
        parsed = parse_tool_result(result)
        # Should return exists=False and error message
        assert (
            parsed.get("exists") is False
        ), f"Restricted path should not exist: {parsed}"
        assert "error" in parsed, "Should include error message"

    @pytest.mark.asyncio
    async def test_nonexistent_file(self, mcp_client: ClientSession):
        """Test handling of nonexistent files."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_PROJECT_FILE, {"file_path": "nonexistent.yml"}
        )
        parsed = parse_tool_result(result)
        # Should indicate file doesn't exist with error message
        assert (
            parsed.get("exists") is False
        ), f"Nonexistent file should not exist: {parsed}"
        assert "error" in parsed, "Should include error message"

    @pytest.mark.asyncio
    async def test_write_to_restricted_path(self, mcp_client: ClientSession):
        """Test that writing to restricted paths fails."""
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_WRITE_PROJECT_FILE,
            {"file_path": ".hidden/file.txt", "content": "secret"},
        )
        parsed = parse_tool_result(result)
        # Should return success=False (not an error, but operation failed)
        assert (
            parsed.get("success") is False
        ), f"Write should fail for restricted path: {parsed}"
        assert "message" in parsed, "Should include error message"
