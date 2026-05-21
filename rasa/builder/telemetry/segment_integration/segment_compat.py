"""Centralized Segment transport for builder telemetry.

All Segment writes from `rasa.builder` should go through this module so the
write-key gate, the telemetry kill-switch, and the failure-swallowing wrapper
live in exactly one place.
"""

import os
from typing import Any, Dict, Optional

import structlog

from rasa import telemetry
from rasa.telemetry import (
    SEGMENT_TRACK_ENDPOINT,
    TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE,
    segment_request_payload,
    send_segment_request,
    with_default_context_fields,
)

structlogger = structlog.get_logger()

_WRITE_KEY: Optional[str] = os.getenv(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE)
_SEGMENT_ON: bool = bool(_WRITE_KEY)

if _SEGMENT_ON:
    structlogger.info("builder.telemetry.enabled")
else:
    structlogger.warning("builder.telemetry.disabled")


def is_segment_enabled() -> bool:
    """Whether Segment events will actually be sent.

    Both the per-app write-key and the global rasa telemetry switch must be on.
    """
    return _SEGMENT_ON and telemetry.is_telemetry_enabled()


def track(event: str, user_id: str, properties: Dict[str, Any]) -> None:
    """Send a Segment track event. No-op when Segment is disabled.

    Telemetry must never break a caller — all transport failures are logged at
    debug/warning level and swallowed.
    """
    if not is_segment_enabled():
        structlogger.debug("builder.telemetry.track.disabled", event=event)
        return

    try:
        payload = segment_request_payload(
            user_id, event, properties, context=with_default_context_fields()
        )
        structlogger.debug("builder.telemetry.track.sending", payload=payload)
        send_segment_request(SEGMENT_TRACK_ENDPOINT, payload, _WRITE_KEY)
    except Exception as e:
        structlogger.warning(
            "builder.telemetry.track_failed", event=event, error=str(e)
        )


def resolve_default_user_id() -> str:
    """Fallback user id for callers without an authenticated context.

    Used e.g. for the MCP stdio transport, where no auth context exists.
    """
    return telemetry.get_telemetry_id() or "unknown"
