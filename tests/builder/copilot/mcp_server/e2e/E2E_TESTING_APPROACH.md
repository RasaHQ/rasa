# End-to-End Testing Approach for MCP Server

## Overview

This document explains the approach for creating and running end-to-end (E2E) tests for the MCP server. These tests verify complete workflows by actually starting the server and connecting to it via an MCP client.

## Architecture

### File Structure

The E2E test suite is organized into the following modules:

- **`conftest.py`**: Pytest fixtures for test setup and teardown
- **`mcp_server_manager.py`**: `MCPServerManager` class for managing MCP server lifecycle
- **`utils.py`**: Utility functions for subprocess management, port finding, and server waiting
- **`constants.py`**: Shared constants including paths and required environment variables
- **`e2e_helper.py`**: Test helper functions for tool execution, result parsing, and assertions
- **`test_e2e_scenarios.py`**: Test scenarios organized by functionality
- **`run_mcp_server.py`**: Script to run MCP server in subprocess
- **`run_rasa_server.py`**: Script to run Rasa server in subprocess
- **`simple_assistant/`**: Minimal Rasa project used as test data

## Test Project Structure

Tests use a minimal valid Rasa project located in `simple_assistant/` that includes:

- `config.yml`: Rasa configuration
- `domain/domain.yml`: Domain definition with intents, responses, and flows
- `data/flows/greet_user_flow.yml`: Flow definitions
- `endpoints.yml`: Endpoint configuration
- `credentials.yml`: Credentials configuration

The `test_project_folder` fixture copies this structure to a temporary directory for each test, ensuring test isolation.

## Test Execution Flow

```
1. Test starts
   ↓
2. `test_project_folder` fixture creates minimal Rasa project
   ↓
3. `mcp_server` fixture starts MCP server in subprocess
   ↓
4. Wait for server to be ready (port check via `wait_for_server`)
   ↓
5. `mcp_client` fixture connects and initializes MCP session
   ↓
6. (Optional) `rasa_server` fixture provides Rasa server for conversation tests
   ↓
7. Test executes tool calls via MCP client (using constants for tool names)
   ↓
8. Test verifies results using helper assertions from `e2e_helper.py`
   ↓
9. Fixtures clean up (stop servers, close connections)
```

## Available Fixtures

All fixtures are defined in `conftest.py`:

- **`test_project_folder`**: Creates a temporary Rasa project by copying `simple_assistant/`
- **`mcp_server`**: Starts MCP server in subprocess, yields `MCPServerManager` instance
- **`mcp_client`**: Creates and initializes MCP client session connected to test server
- **`rasa_server`**: Provides Rasa server for conversation tests (yields `(port, start_function)` tuple)
- **`mcp_server_port`**: Returns a free port for parametrized tests

## Environment Variables

The test suite requires certain environment variables to be set, depending on which tests you're running:

- **`RASA_PRO_LICENSE`**: Required for validation and training tests (`TestE2EValidationWorkflow`, `TestE2EAssistantConversation`)
- **`OPENAI_API_KEY`**: Required for training tests (`TestE2EAssistantConversation`)
- **`INKEEP_API_KEY`**: Required for documentation search tests (`TestE2EDocumentationSearch`)

These environment variables are automatically passed to test subprocesses via the `prepare_test_env` function in `utils.py`. If a required variable is missing, the corresponding tests will fail.

## Running Tests

### Basic Commands

```bash
# Run all E2E tests
pytest tests/builder/copilot/mcp_server/e2e/test_e2e_scenarios.py -v

# Run with verbose output and print statements
pytest tests/builder/copilot/mcp_server/e2e/test_e2e_scenarios.py -vv -s

# Run specific test class
pytest tests/builder/copilot/mcp_server/e2e/test_e2e_scenarios.py::TestE2EFileOperations -v
```

### Debugging Server Output

By default, MCP and Rasa server output (stdout/stderr) is captured and only shown on errors. To see all server output in real-time for debugging, set the `DEBUG_SERVER_OUTPUT` environment variable:

```bash
# Enable server output logging
DEBUG_SERVER_OUTPUT=1 pytest tests/builder/copilot/mcp_server/e2e/test_e2e_scenarios.py -vv -s
```

When enabled, you'll see output prefixed with:

- `[STDOUT] MCP Server: ...` - MCP server stdout
- `[LOG] MCP Server: ...` - MCP server stderr (logging output, not necessarily errors)
- `[STDOUT] Rasa Server: ...` - Rasa server stdout
- `[LOG] Rasa Server: ...` - Rasa server stderr (logging output, not necessarily errors)

Note: Many applications use stderr for all logging output, not just errors, so the `[LOG]` prefix indicates logging information rather than actual errors.

This is especially useful when debugging server startup issues, connection problems, or tool execution errors.

The logging is handled by `start_logging_threads` in `utils.py`, which always consumes subprocess output to prevent pipe buffer blocking, but only prints when `DEBUG_SERVER_OUTPUT` is enabled.

### Debugging MCP Responses

To inspect the raw responses from the MCP server (before parsing), set the `DEBUG_MCP_RESPONSE` environment variable:

```bash
# Enable MCP response debugging
DEBUG_MCP_RESPONSE=1 pytest tests/builder/copilot/mcp_server/e2e/test_e2e_scenarios.py -vv -s
```

When enabled, you'll see detailed output for each tool call showing:

- `isError`: Whether the response indicates an error
- `structuredContent`: The structured content (if any) returned by the tool
- `content`: The raw content items (TextContent, etc.)
- Content details: Type information and text for each content item

Example output:

```
================================================================================
[DEBUG] Raw MCP Response for tool: train_model
================================================================================
isError: False
structuredContent: {'success': True, 'model_path': '/path/to/model.tar.gz', ...}
content: [TextContent(type='text', text='{...}')]

Content details:
  Content item 0:
    Type: <class 'mcp.types.TextContent'>
    Text: {"success": true, "model_path": "...", ...}
    Type (content): text
================================================================================
```

This is especially useful when:

- Debugging response parsing issues
- Understanding the structure of tool responses
- Verifying that assertions match the actual response format
- Troubleshooting why tests fail unexpectedly
