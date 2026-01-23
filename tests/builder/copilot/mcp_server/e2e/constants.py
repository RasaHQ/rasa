"""Constants for MCP server end-to-end tests."""

from pathlib import Path

# Base directory for e2e tests
BASE_MCP_E2E_TEST_DIR = Path(__file__).parent

# Paths to test scripts
RUN_MCP_SERVER_SCRIPT_PATH = BASE_MCP_E2E_TEST_DIR / "run_mcp_server.py"
RUN_RASA_SERVER_SCRIPT_PATH = BASE_MCP_E2E_TEST_DIR / "run_rasa_server.py"

# Path to simple_assistant test project
SIMPLE_ASSISTANT_PATH = BASE_MCP_E2E_TEST_DIR / "simple_assistant"

# Required environment variables for test subprocesses
# - RASA_PRO_LICENSE: Required for validation and training tests
# - OPENAI_API_KEY: Required for training tests
# - INKEEP_API_KEY: Required for documentation search tests (TestE2EDocumentationSearch)
REQUIRED_ENV_VARS = ["RASA_PRO_LICENSE", "OPENAI_API_KEY", "INKEEP_API_KEY"]
