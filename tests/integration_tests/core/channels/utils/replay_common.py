"""
Shared utilities for voice channel WebSocket replay tests.

Used by websocket_replay (AudioCodes), twilio_media_stream_replay,
and jambonz_replay to avoid duplicating common logic.
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NORMAL_CLOSE_CODE = 1000


def load_traffic_log(log_file: str) -> Optional[List[Dict[str, Any]]]:
    """Load a traffic log from a JSON file. Returns None on error and logs."""
    try:
        with open(log_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("File not found: %s", log_file)
        return None
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in log file: %s", e)
        return None


def build_websocket_url(rasa_url: str, path_suffix: str) -> str:
    """Build WebSocket URL from Rasa base URL and path (no leading slash)."""
    base = rasa_url.replace("http://", "ws://").replace("https://", "wss://")
    return f"{base.rstrip('/')}/{path_suffix.lstrip('/')}"


def is_graceful_connection_error(error: Exception) -> bool:
    """Return True if the exception indicates a graceful connection closure."""
    error_str = str(error).lower()
    error_type = type(error).__name__
    return (
        "closing transport" in error_str
        or "connection closed" in error_str
        or error_type in ("ConnectionResetError", "ClientConnectionResetError")
    )


def record_connection_closed_before_send(
    errors: List[Dict[str, Any]], message_index: int, message_kind: str = "unknown"
) -> None:
    """Append a 'connection closed before sending' error to the errors list."""
    errors.append(
        {
            "message_index": message_index,
            "message_kind": message_kind,
            "error": "Connection closed before sending",
        }
    )


def record_send_error(
    errors: List[Dict[str, Any]],
    message_index: int,
    message_kind: str,
    error: Exception,
) -> None:
    """Append a send error to the errors list with graceful flag."""
    errors.append(
        {
            "message_index": message_index,
            "message_kind": message_kind,
            "error": str(error),
            "graceful": is_graceful_connection_error(error),
        }
    )


def record_server_close(
    errors: List[Dict[str, Any]],
    message_index: int,
    message_kind: str,
    close_code: Any,
) -> None:
    """Append a server-close error to the errors list."""
    errors.append(
        {
            "message_index": message_index,
            "message_kind": message_kind,
            "error": f"Connection closed by server (code: {close_code})",
            "graceful": close_code == NORMAL_CLOSE_CODE,
        }
    )


def record_outer_error(errors: List[Dict[str, Any]], error: Exception) -> None:
    """Append an outer WebSocket/session error to the errors list."""
    errors.append(
        {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "graceful": is_graceful_connection_error(error),
        }
    )


def print_session_summary(
    connection_state: Dict[str, Any],
    state_keys: Optional[List[tuple]] = None,
) -> None:
    """Print a generic session summary. state_keys: list of (key, label) to print."""
    logger.info("\n" + "=" * 60)
    logger.info("SESSION SUMMARY")
    logger.info("=" * 60)
    logger.debug(
        "Sent count: %s",
        connection_state.get("messages_sent", 0),
    )
    logger.debug(
        "Recv count: %s",
        connection_state.get("messages_received", 0),
    )
    if state_keys:
        for key, label in state_keys:
            if key in connection_state:
                logger.info("%s: %s", label, connection_state[key])
    errors = connection_state.get("errors", [])
    if errors:
        graceful = [e for e in errors if e.get("graceful", False)]
        actual = [e for e in errors if not e.get("graceful", False)]
        if graceful:
            logger.info("\nGraceful connection closures: %s", len(graceful))
        if actual:
            logger.warning("\nErrors encountered: %s", len(actual))
        elif not graceful:
            logger.info("\nNo errors encountered")
    else:
        logger.info("\nNo errors encountered")
    logger.info("=" * 60 + "\n")
