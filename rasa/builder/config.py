"""Configuration module for the prompt-to-bot service."""

import os
from typing import Any, Dict

import importlib_resources

from rasa.constants import PACKAGE_NAME
from rasa.shared.utils.yaml import read_yaml, read_yaml_file

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-2025-04-14")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
OPENAI_VECTOR_STORE_ID = os.getenv(
    "OPENAI_VECTOR_STORE_ID", "vs_685123376e288191a005b6b144d3026f"
)
OPENAI_MAX_VECTOR_RESULTS = int(os.getenv("OPENAI_MAX_VECTOR_RESULTS", "10"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "30"))

# Server Configuration
BUILDER_SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
BUILDER_SERVER_PORT = int(os.getenv("SERVER_PORT", "5050"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MAX_LOG_ENTRIES = int(os.getenv("MAX_LOG_ENTRIES", "30"))
HELLO_RASA_PROJECT_ID = os.getenv("HELLO_RASA_PROJECT_ID")

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

# Guardrail Configuration
LAKERA_BASE_URL = os.getenv("LAKERA_BASE_URL", "https://api.lakera.ai/v2").rstrip("/")
ASSISTANT_HISTORY_GUARDRAIL_PROJECT_ID = "project-5442154580"
COPILOT_HISTORY_GUARDRAIL_PROJECT_ID = "project-4637497223"

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

# Compute proxy-aware base URLs
if HELLO_LLM_PROXY_BASE_URL:
    _proxy = HELLO_LLM_PROXY_BASE_URL.rstrip("/")
    INKEEP_BASE_URL = f"{_proxy}/documentation"
    LAKERA_BASE_URL = f"{_proxy}/guardrails"


# Number of minutes after FIRST_USED when authentication becomes required
# across routes (except when auth is globally disabled on the server).
# Default kept small for development; adjust as needed.
AUTH_REQUIRED_AFTER_MINUTES = int(os.getenv("AUTH_REQUIRED_AFTER_MINUTES", "480"))


def get_default_config(assistant_id: str) -> Dict[str, Any]:
    """Get default Rasa configuration."""
    base_config = read_yaml_file(
        str(
            importlib_resources.files(PACKAGE_NAME).joinpath(
                "cli/project_templates/default/config.yml"
            )
        )
    )

    if not isinstance(base_config, dict):
        raise ValueError("Base config is not a dictionary")

    base_config["assistant_id"] = assistant_id

    return base_config


def get_default_endpoints() -> Dict[str, Any]:
    """Get default endpoints configuration."""
    endpoints_config = read_yaml_file(
        str(
            importlib_resources.files(PACKAGE_NAME).joinpath(
                "cli/project_templates/default/endpoints.yml"
            )
        )
    )

    if not isinstance(endpoints_config, dict):
        raise ValueError("Endpoints config is not a dictionary")

    return endpoints_config


def get_default_credentials() -> Dict[str, Any]:
    """Get default credentials configuration."""
    default_credentials_yaml = """
    studio_chat:
      user_message_evt: "user_message"
      bot_message_evt: "bot_message"
      session_persistence: true
    """
    return read_yaml(default_credentials_yaml)
