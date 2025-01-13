from __future__ import annotations

import enum
from typing import Dict
from urllib.parse import urlparse

import structlog

structlogger = structlog.get_logger()


class UrlSchema(enum.Enum):
    HTTP = "http"
    HTTPS = "https"
    FILE = "file"
    FTP = "ftp"
    SFTP = "sftp"
    GRPC = "grpc"
    UNKNOWN = "unknown"
    NOT_SPECIFIED = "not_specified"

    @property
    def available_schemas(self) -> Dict[str, UrlSchema]:
        """Get all available URL schemas except for the unknown schema."""
        return {
            schema.value: schema for schema in UrlSchema if schema != UrlSchema.UNKNOWN
        }


def get_url_schema(url: str) -> UrlSchema:
    """Get the schema of a URL.

    Args:
        url: The URL to parse.

    Returns:
        The schema of the URL.
    """
    parsed_url = urlparse(url)

    if parsed_url.scheme == "":
        return UrlSchema.NOT_SPECIFIED

    try:
        return UrlSchema(parsed_url.scheme)
    except ValueError:
        structlogger.warn(
            "rasa.utils.url_tools.get_url_schema.unknown_schema",
            event_info=f"Unknown URL schema '{parsed_url.scheme}'. "
            f"Returning 'unknown'.",
        )

        return UrlSchema.UNKNOWN
