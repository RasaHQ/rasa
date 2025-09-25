"""Fundamental types for authentication protocols."""

from enum import Enum


class AgentAuthType(Enum):
    """An Enum class that represents the supported authentication protocol types."""

    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BEARER_TOKEN = "bearer_token"
    CUSTOM = "custom"
