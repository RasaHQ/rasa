"""Constants for MCP server."""

from pathlib import Path

INSTRUCTIONS_FILE_PATH = Path(__file__).parent / "INSTRUCTIONS.md"

# MCP Tool Names - Centralized constants for tool names used throughout the codebase
# These are Rasa-specific tools exposed via the MCP server
MCP_TOOL_SEARCH_RASA_DOCS = "search_rasa_documentation"
MCP_TOOL_VALIDATE_PROJECT = "validate_project"
MCP_TOOL_TRAIN_RASA_ASSISTANT = "train_rasa_assistant"
MCP_TOOL_GET_ASSISTANT_LOGS = "get_assistant_logs"
MCP_TOOL_TALK_TO_ASSISTANT = "talk_to_assistant"
MCP_TOOL_LIST_CUSTOM_ACTION_IMPLEMENTATIONS = "list_custom_action_implementations"

# Project context tools - for listing flows, slots, responses, and custom actions
MCP_TOOL_LIST_FLOWS = "list_project_flow_definitions"
MCP_TOOL_LIST_SLOTS = "list_project_slot_definitions"
MCP_TOOL_LIST_RESPONSES = "list_project_response_definitions"
MCP_TOOL_LIST_DOMAIN_ACTIONS = "list_project_custom_actions_in_domain"
MCP_TOOL_LIST_DEFAULT_ACTIONS = "list_default_action_names"

# Granular getters - retrieve single flow/slot/response by identifier
MCP_TOOL_GET_FLOW = "get_flow"
MCP_TOOL_GET_SLOT = "get_slot"
MCP_TOOL_GET_RESPONSE = "get_response"

# Schema tools - expose Rasa schemas for validation and code generation
MCP_TOOL_GET_FLOW_SCHEMA = "get_flow_schema"
MCP_TOOL_GET_DOMAIN_SCHEMA = "get_domain_schema"
MCP_TOOL_GET_E2E_SCHEMA = "get_e2e_schema"
