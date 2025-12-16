"""Configuration module for the prompt-to-bot service."""

import os
from typing import Any, Dict

from rasa.shared.utils.yaml import read_yaml

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_VECTOR_STORE_ID = os.getenv(
    "OPENAI_VECTOR_STORE_ID", "vs_685123376e288191a005b6b144d3026f"
)
OPENAI_MAX_VECTOR_RESULTS = int(os.getenv("OPENAI_MAX_VECTOR_RESULTS", "10"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))

# Project generation
PROJECT_GENERATION_TIMEOUT = int(os.getenv("PROJECT_GENERATION_TIMEOUT", "1000"))
PROJECT_GENERATION_MAX_RETRIES = int(os.getenv("PROJECT_GENERATION_MAX_RETRIES", "5"))

# OpenAI Token Pricing Configuration (per 1,000 tokens)
COPILOT_INPUT_TOKEN_PRICE = float(os.getenv("COPILOT_INPUT_TOKEN_PRICE", "0.002"))
COPILOT_OUTPUT_TOKEN_PRICE = float(os.getenv("COPILOT_OUTPUT_TOKEN_PRICE", "0.0005"))
COPILOT_CACHED_TOKEN_PRICE = float(os.getenv("COPILOT_CACHED_TOKEN_PRICE", "0.002"))

# Server Configuration
BUILDER_SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
BUILDER_SERVER_PORT = int(os.getenv("SERVER_PORT", "5050"))
MAX_LOG_ENTRIES = int(os.getenv("MAX_LOG_ENTRIES", "30"))
HELLO_RASA_PROJECT_ID = os.getenv("HELLO_RASA_PROJECT_ID")


# Copilot History Storage Configuration
# Relative path from project root to copilot database
COPILOT_DB_RELATIVE_PATH = ".rasa/copilot.db"
# Absolute path to copilot database (relative to current working directory)
# TODO: this shouldn't exist and should be removed. All data relevant to a project
# should be stored in the project folder. We can't do that at the moment, because
# we have a global llmservice singelton that is independent of the project folder.
# we should move away from that singleton (or at least extract the project specific
# data from the singleton to a separate service).
COPILOT_HISTORY_SQLITE_PATH = os.path.join(os.getcwd(), COPILOT_DB_RELATIVE_PATH)

# CORS Configuration
_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = _cors_origins_env.split(",") if _cors_origins_env != "*" else ["*"]

# Validation Configuration
VALIDATION_FAIL_ON_WARNINGS = (
    os.getenv("VALIDATION_FAIL_ON_WARNINGS", "false").lower() == "true"
)
VALIDATION_MAX_HISTORY = None  # Could be configured if needed

# Copilot Response Handler Configuration
COPILOT_CONTROLLED_RESPONSE_MAX_TOKENS = 20
COPILOT_HANDLER_ROLLING_BUFFER_SIZE = 20
COPILOT_ASSISTANT_TRACKER_MAX_TURNS = 10
COPILOT_DOCUMENTATION_SEARCH_QUERY_HISTORY_MESSAGES = 5

# Guardrail Configuration
ENABLE_GUARDRAILS_RAW = os.getenv("ENABLE_GUARDRAILS", "false")
ENABLE_GUARDRAILS = ENABLE_GUARDRAILS_RAW.strip().lower() == "true"

# TODO: Replace with Open Source guardrails implementation once it's ready
LAKERA_BASE_URL = os.getenv("LAKERA_BASE_URL", "https://api.lakera.ai/v2").rstrip("/")
LAKERA_ASSISTANT_HISTORY_GUARDRAIL_PROJECT_ID = "project-5442154580"
LAKERA_COPILOT_HISTORY_GUARDRAIL_PROJECT_ID = "project-4637497223"

# Guardrails blocking configuration
GUARDRAILS_ENABLE_BLOCKING_RAW = os.getenv("GUARDRAILS_ENABLE_BLOCKING", "false")
GUARDRAILS_ENABLE_BLOCKING = GUARDRAILS_ENABLE_BLOCKING_RAW.strip().lower() == "true"
GUARDRAILS_USER_MAX_STRIKES = int(os.getenv("GUARDRAILS_USER_MAX_STRIKES", "3"))
GUARDRAILS_PROJECT_MAX_STRIKES = int(os.getenv("GUARDRAILS_PROJECT_MAX_STRIKES", "5"))
GUARDRAILS_BLOCK_DURATION_SECONDS = int(
    os.getenv("GUARDRAILS_BLOCK_DURATION_SECONDS", "0")
)  # 0 means indefinite

# Auth0 Configuration
AUTH0_DOMAIN = "login.hello.rasa.ai"
AUTH0_CLIENT_ID = "Gq5RdRwp174OFIfTz6Re9TZUseYDXUYE"
AUTH0_ISSUER = f"https://{AUTH0_DOMAIN}/"
JWKS_URL = f"{AUTH0_ISSUER}.well-known/jwks.json"

# Inkeep Configuration
INKEEP_BASE_URL = os.getenv("INKEEP_BASE_URL", "https://api.inkeep.com/v1").rstrip("/")

# LLM Proxy Configuration
HELLO_LLM_PROXY_BASE_URL = os.getenv("HELLO_LLM_PROXY_BASE_URL")
RASA_PRO_LICENSE = os.getenv("RASA_PRO_LICENSE")

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_DEFAULT_ENVIRONMENT = "default"
DEPLOYMENT_STACK = os.getenv("DEPLOYMENT_STACK", "local")
DEPLOYMENT_STACK_HEADER_NAME = "deployment-stack"

# Compute proxy-aware base URLs
if HELLO_LLM_PROXY_BASE_URL:
    _proxy = HELLO_LLM_PROXY_BASE_URL.rstrip("/")
    INKEEP_BASE_URL = f"{_proxy}/documentation"
    LAKERA_BASE_URL = f"{_proxy}/guardrails"
    LANGFUSE_HOST = f"{_proxy}/langfuse"


# Number of minutes after FIRST_USED when authentication becomes required
# across routes (except when auth is globally disabled on the server).
# Default kept small for development; adjust as needed.
AUTH_REQUIRED_AFTER_MINUTES = int(os.getenv("AUTH_REQUIRED_AFTER_MINUTES", "480"))

DEFAULT_BOT_BUILDER_EMAIL = "noreply@rasa.com"

# Maximum characters/tokens for commit messages
CHARS_PER_TOKEN_ESTIMATE = 4
# Desired length for commit messages in prompting
COMMIT_MESSAGE_DESIRED_LENGTH = int(
    os.getenv("COMMIT_MESSAGE_LENGTH_WARNING_THRESHOLD", "42")
)
# Recommended max characters for commit messages
COMMIT_MESSAGE_MAX_CHARACTERS = int(os.getenv("COMMIT_MESSAGE_MAX_CHARACTERS", "72"))
# Calculate max tokens based on max characters
COMMIT_MESSAGE_MAX_TOKENS = int(
    os.getenv(
        "COMMIT_MESSAGE_MAX_TOKENS",
        COMMIT_MESSAGE_MAX_CHARACTERS // CHARS_PER_TOKEN_ESTIMATE,
    )
)

# Maximum tokens for welcome messages
WELCOME_MESSAGE_MAX_TOKENS = int(os.getenv("WELCOME_MESSAGE_MAX_TOKENS", "50"))

# Agent SDK Configuration
USE_AGENT_SDK_COPILOT = os.getenv("USE_AGENT_SDK_COPILOT", "false").lower() == "true"


# MCP Server Configuration
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "5051"))
# Timeout for MCP server client session (in seconds)
# This sets client_session_timeout_seconds for MCPServer
# Validation and training tools can take 60+ seconds
MCP_TOOL_CALL_TIMEOUT = int(os.getenv("MCP_TOOL_CALL_TIMEOUT", "120"))
MCP_MAX_RETRY_ATTEMPTS = int(os.getenv("MCP_MAX_RETRY_ATTEMPTS", "3"))
# Timeout for waiting for MCP server to start accepting connections (in seconds)
MCP_SERVER_STARTUP_TIMEOUT = int(os.getenv("MCP_SERVER_STARTUP_TIMEOUT", "30"))


def get_default_credentials() -> Dict[str, Any]:
    """Get default credentials configuration."""
    default_credentials_yaml = """
    studio_chat:
      user_message_evt: "user_message"
      bot_message_evt: "bot_message"
      session_persistence: true
    """
    return read_yaml(default_credentials_yaml)
