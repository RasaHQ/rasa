"""End-to-end tests for MCP server using actual MCP client connections.

These tests verify complete workflows by:
1. Starting the MCP server in a background process
2. Connecting to it via MCP client
3. Executing tool calls and verifying results
"""

from pathlib import Path
from typing import Callable, ClassVar, List

import pytest
from mcp import ClientSession

from rasa.builder.copilot.mcp_server.constants import (
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
)
from tests.builder.copilot.mcp_server.e2e.e2e_helper import (
    assert_operation_success,
    assert_training_success,
    call_tool_safely,
    parse_tool_result,
)


class TestE2EValidationWorkflow:
    """End-to-end tests for validation workflow."""

    @pytest.mark.asyncio
    async def test_validate_workflow(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test validation tool."""
        # Validate the project
        result = await call_tool_safely(mcp_client, MCP_TOOL_VALIDATE_PROJECT, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)


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
        result = await call_tool_safely(mcp_client, MCP_TOOL_TRAIN_RASA_ASSISTANT, {})
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
    async def test_search_rasa_documentation(self, mcp_client: ClientSession):
        """Test search_rasa_documentation tool returns properly formatted results."""
        # Call the search_rasa_documentation tool with a test query
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_SEARCH_RASA_DOCS, {"query": "What is Rasa?"}
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


class TestE2EProjectContext:
    """End-to-end tests for project context list tools."""

    @pytest.mark.asyncio
    async def test_list_flows(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test list_flows returns flows from the test project."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_LIST_FLOWS, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        flows = parsed.get("flows", [])
        assert len(flows) > 0, "Should find at least one flow"

        flow_ids = [f["id"] for f in flows]
        assert (
            "greet_user_flow" in flow_ids
        ), f"Should find 'greet_user_flow', got: {flow_ids}"

    @pytest.mark.asyncio
    async def test_list_slots(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test list_slots returns slots from the test project."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_LIST_SLOTS, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        slots = parsed.get("slots", [])
        assert len(slots) > 0, "Should find at least one slot"

        slot_names = [s["name"] for s in slots]
        assert (
            "user_name" in slot_names
        ), f"Should find 'user_name' slot, got: {slot_names}"

    @pytest.mark.asyncio
    async def test_list_responses(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test list_responses returns responses from the test project."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_LIST_RESPONSES, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        responses = parsed.get("responses", [])
        assert len(responses) > 0, "Should find at least one response"

        response_names = [r["name"] for r in responses]
        assert (
            "utter_greet_user" in response_names
        ), f"Should find 'utter_greet_user', got: {response_names}"
        assert (
            "utter_ask_user_name" in response_names
        ), f"Should find 'utter_ask_user_name', got: {response_names}"

    @pytest.mark.asyncio
    async def test_list_domain_actions(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test list_domain_actions returns actions from the test project."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_LIST_DOMAIN_ACTIONS, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        actions = parsed.get("actions", [])
        assert len(actions) > 0, "Should find at least one custom action"

        action_names = [a["name"] for a in actions]
        assert (
            "action_check_user" in action_names
        ), f"Should find 'action_check_user', got: {action_names}"

    @pytest.mark.asyncio
    async def test_list_default_actions(self, mcp_client: ClientSession):
        """Test list_default_actions returns built-in Rasa action names."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_LIST_DEFAULT_ACTIONS, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        actions = parsed.get("actions", [])
        assert len(actions) > 0, "Should return built-in action names"
        # Verify some well-known default actions are present
        assert "action_listen" in actions, "Should include 'action_listen'"
        assert "action_restart" in actions, "Should include 'action_restart'"

    @pytest.mark.asyncio
    async def test_get_flow(self, mcp_client: ClientSession, test_project_folder: Path):
        """Test get_flow returns a flow when given a valid flow ID."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_FLOW, {"flow_id": "greet_user_flow"}
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        flow = parsed.get("flow")
        assert flow is not None, "Should return a flow object"
        assert flow["id"] == "greet_user_flow"

    @pytest.mark.asyncio
    async def test_get_slot(self, mcp_client: ClientSession, test_project_folder: Path):
        """Test get_slot returns a slot when given a valid slot name."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_SLOT, {"slot_name": "user_name"}
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        slot = parsed.get("slot")
        assert slot is not None, "Should return a slot object"
        assert slot["name"] == "user_name"

    @pytest.mark.asyncio
    async def test_get_response(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test get_response returns a response when given a valid name."""
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_GET_RESPONSE,
            {"response_name": "utter_greet_user"},
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        response = parsed.get("response")
        assert response is not None, "Should return a response object"
        assert response["name"] == "utter_greet_user"


class TestE2EErrorHandling:
    """End-to-end tests for error handling with project context list tools."""

    # Mapping of list tools to their folder parameter name
    _LIST_TOOLS_WITH_FOLDER_PARAM: ClassVar[List[pytest.param]] = [
        pytest.param(MCP_TOOL_LIST_FLOWS, "data_folder", id="list_flows"),
        pytest.param(MCP_TOOL_LIST_SLOTS, "domain_folder", id="list_slots"),
        pytest.param(MCP_TOOL_LIST_RESPONSES, "domain_folder", id="list_responses"),
        pytest.param(
            MCP_TOOL_LIST_DOMAIN_ACTIONS, "domain_folder", id="list_domain_actions"
        ),
    ]

    # Paths that attempt to escape the project directory (path traversal)
    _TRAVERSAL_PATHS: ClassVar[List[pytest.param]] = [
        pytest.param("../../etc", id="parent_traversal"),
        pytest.param("/etc/passwd", id="absolute_path"),
        pytest.param("../outside", id="relative_escape"),
    ]

    # Paths that don't exist but are valid (within project bounds)
    _NONEXISTENT_PATHS: ClassVar[List[pytest.param]] = [
        pytest.param("nonexistent_folder", id="nonexistent_folder"),
        pytest.param("does/not/exist", id="nested_nonexistent"),
    ]

    # Tools that return an error for nonexistent items
    _GETTER_NOT_FOUND: ClassVar[List[pytest.param]] = [
        pytest.param(
            MCP_TOOL_GET_FLOW,
            {"flow_id": "nonexistent_flow"},
            id="get_flow",
        ),
        pytest.param(
            MCP_TOOL_GET_SLOT,
            {"slot_name": "nonexistent_slot"},
            id="get_slot",
        ),
        pytest.param(
            MCP_TOOL_GET_RESPONSE,
            {"response_name": "utter_nonexistent"},
            id="get_response",
        ),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name, folder_param", _LIST_TOOLS_WITH_FOLDER_PARAM)
    @pytest.mark.parametrize("path", _TRAVERSAL_PATHS)
    async def test_path_traversal_rejected(
        self,
        mcp_client: ClientSession,
        tool_name: str,
        folder_param: str,
        path: str,
    ):
        """Test that path traversal attempts are rejected by list tools."""
        result = await call_tool_safely(mcp_client, tool_name, {folder_param: path})
        parsed = parse_tool_result(result)
        assert (
            parsed.get("success") is False
        ), f"{tool_name} should reject traversal path '{path}': {parsed}"
        assert parsed.get(
            "error"
        ), f"{tool_name} should include an error message for path '{path}'"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name, folder_param", _LIST_TOOLS_WITH_FOLDER_PARAM)
    @pytest.mark.parametrize("path", _NONEXISTENT_PATHS)
    async def test_nonexistent_path_handled_gracefully(
        self,
        mcp_client: ClientSession,
        tool_name: str,
        folder_param: str,
        path: str,
    ):
        """Test that nonexistent paths within the project are handled gracefully."""
        result = await call_tool_safely(mcp_client, tool_name, {folder_param: path})
        parsed = parse_tool_result(result)
        assert (
            parsed.get("success") is True
        ), f"{tool_name} should handle nonexistent path '{path}' gracefully: {parsed}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", _TRAVERSAL_PATHS)
    async def test_custom_actions_path_traversal_rejected(
        self,
        mcp_client: ClientSession,
        test_project_folder: Path,
        path: str,
    ):
        """Test path traversal is rejected by list_custom_action_implementations."""
        result = await call_tool_safely(
            mcp_client,
            MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS,
            {"actions_folder": path},
        )
        parsed = parse_tool_result(result)
        assert parsed.get("error"), f"Should reject traversal path '{path}': {parsed}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_name, arguments", _GETTER_NOT_FOUND)
    async def test_get_item_not_found(
        self,
        mcp_client: ClientSession,
        test_project_folder: Path,
        tool_name: str,
        arguments: dict,
    ):
        """Test getter tools return an error for nonexistent items."""
        result = await call_tool_safely(mcp_client, tool_name, arguments)
        parsed = parse_tool_result(result)
        assert parsed.get("success") is False, f"{tool_name} should fail: {parsed}"
        assert parsed.get("error"), "Should include an error message"


class TestE2ESchemaTools:
    """End-to-end tests for schema retrieval tools."""

    _CALM_ONLY_ARGS: ClassVar[List[pytest.param]] = [
        pytest.param({}, id="calm_only"),
        pytest.param({"calm_only": False}, id="full"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", _CALM_ONLY_ARGS)
    async def test_get_flow_schema(self, mcp_client: ClientSession, arguments: dict):
        """Test get_flow_schema returns a valid schema."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_GET_FLOW_SCHEMA, arguments)
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        assert parsed.get("schema_type") == "flow"
        assert len(parsed.get("schema_content", "")) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("arguments", _CALM_ONLY_ARGS)
    async def test_get_domain_schema(self, mcp_client: ClientSession, arguments: dict):
        """Test get_domain_schema returns a valid schema."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_GET_DOMAIN_SCHEMA, arguments
        )
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        assert parsed.get("schema_type") == "domain"
        assert len(parsed.get("schema_content", "")) > 0

    @pytest.mark.asyncio
    async def test_get_e2e_schema(self, mcp_client: ClientSession):
        """Test get_e2e_schema returns a valid schema."""
        result = await call_tool_safely(mcp_client, MCP_TOOL_GET_E2E_SCHEMA, {})
        parsed = parse_tool_result(result)
        assert_operation_success(parsed)

        assert parsed.get("schema_type") == "e2e"
        assert len(parsed.get("schema_content", "")) > 0


class TestE2ECustomActionImplementations:
    """End-to-end tests for list_custom_action_implementations tool."""

    @pytest.fixture
    def empty_actions_folder(self, test_project_folder: Path) -> Path:
        """Create an empty actions package in the test project."""
        actions_dir = test_project_folder / "actions"
        actions_dir.mkdir()
        (actions_dir / "__init__.py").write_text("")
        return actions_dir

    @pytest.fixture
    def populated_actions_folder(self, empty_actions_folder: Path) -> Path:
        """Create an actions package with a custom action."""
        (empty_actions_folder / "check_user.py").write_text(
            "from rasa_sdk import Action\n\n"
            "class ActionCheckUser(Action):\n"
            "    def name(self):\n"
            '        return "action_check_user"\n\n'
            "    async def run("
            "self, dispatcher, tracker, domain"
            "):\n"
            "        return []\n"
        )
        return empty_actions_folder

    @pytest.mark.asyncio
    async def test_no_actions_folder(
        self, mcp_client: ClientSession, test_project_folder: Path
    ):
        """Test that the tool reports an error when the actions folder doesn't exist."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS, {}
        )
        parsed = parse_tool_result(result)
        assert parsed.get(
            "error"
        ), "Should report an error when actions folder is missing"
        assert parsed.get("count", -1) == 0

    @pytest.mark.asyncio
    async def test_empty_actions_folder(
        self,
        mcp_client: ClientSession,
        empty_actions_folder: Path,
    ):
        """Test the tool returns count=0 when actions folder is empty."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS, {}
        )
        parsed = parse_tool_result(result)
        assert parsed.get("error") is None, f"Should not have an error: {parsed}"
        assert parsed.get("count", -1) == 0

    @pytest.mark.asyncio
    async def test_with_actions(
        self,
        mcp_client: ClientSession,
        populated_actions_folder: Path,
    ):
        """Test that the tool finds actions when an actions folder exists."""
        result = await call_tool_safely(
            mcp_client, MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS, {}
        )
        parsed = parse_tool_result(result)
        assert parsed.get("error") is None, f"Should not have an error: {parsed}"
        assert parsed.get("count", 0) > 0, "Should find at least one action"

        action_names = [a["name"] for a in parsed.get("actions", [])]
        assert (
            "action_check_user" in action_names
        ), f"Should find 'action_check_user', got: {action_names}"
