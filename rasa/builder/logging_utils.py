"""Logging and Sentry utilities for the builder service."""

import collections
import contextvars
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Deque, Dict, Generator, List, Mapping, MutableMapping, Optional

import sentry_sdk
import structlog
from sanic import Request

from rasa.builder import config
from rasa.builder.auth import HEADER_USER_ID

structlogger = structlog.get_logger()

# Thread-safe deque for collecting recent logs
_recent_logs: Deque[str] = collections.deque(maxlen=config.MAX_LOG_ENTRIES)
_logs_lock = threading.RLock()
# Context variable for validation logs (async-safe)
_validation_logs: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = (
    contextvars.ContextVar("validation_logs", default=None)
)


def collecting_logs_processor(
    logger: Any, log_level: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that collects recent log entries.

    This processor is thread-safe and maintains a rolling buffer of recent logs.
    """
    if log_level != logging.getLevelName(logging.DEBUG).lower():
        event_message = event_dict.get("event_info") or event_dict.get("event", "")
        log_entry = f"[{log_level}] {event_message}"

        with _logs_lock:
            _recent_logs.append(log_entry)

    return event_dict


def collecting_validation_logs_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that captures validation logs in context variable storage.

    It's designed to be used with the capture_validation_logs context manager.
    Uses contextvars for async-safe log capture across async tasks.

    Args:
        logger: The structlog logger instance
        method_name: The logging method name (e.g., "error", "warning", "info", "debug")
        event_dict: The event dictionary containing log data

    Returns:
        The unmodified event_dict (this processor doesn't modify the log data)
    """
    # Only capture logs if we're in a validation context
    # (logs list exists in the current context)
    logs = _validation_logs.get()
    if logs is not None:
        log_entry = {"log_level": method_name, **event_dict}
        logs.append(log_entry)

    return event_dict


@contextmanager
def capture_validation_logs() -> Generator[List[Dict[str, Any]], Any, None]:
    """Context manager to capture validation logs using context variables.

    This context manager stores logs in a context variable WITHOUT reconfiguring
    structlog globally. The processor checks the context variable and captures
    logs if present. This avoids race conditions with concurrent requests.

    Yields:
        A list of captured log entries, each containing the log level and all
        original log data from the event_dict.
    """
    # Initialize context variable logs storage
    # The processor is ALWAYS installed (see module init), it just checks
    # this context var
    logs: List[Dict[str, Any]] = []
    token = _validation_logs.set(logs)

    try:
        yield logs
    finally:
        # Clean up context variable
        _validation_logs.reset(token)


def attach_request_id_processor(
    logger: Any, log_level: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Structlog processor that attaches the request id to the event dict.

    This processor is thread-safe and maintains a rolling buffer of recent logs.
    """
    try:
        request = Request.get_current()
        event_dict["correlation_id"] = request.ctx.correlation_id
        return event_dict
    except Exception:
        # there is no request context, so we don't attach the request id
        return event_dict


def get_recent_logs() -> str:
    """Get recent log entries as a formatted string.

    Returns:
        Formatted string of recent log entries, one per line.
    """
    with _logs_lock:
        return "\n".join(list(_recent_logs))


def clear_recent_logs() -> None:
    """Clear the recent logs buffer."""
    with _logs_lock:
        _recent_logs.clear()


def get_log_count() -> int:
    """Get the current number of log entries."""
    with _logs_lock:
        return len(_recent_logs)


def _sanitize_headers(headers: Mapping[str, str]) -> Dict[str, Any]:
    """Remove or redact sensitive headers for safe logging and Sentry context."""
    lowered = {k.lower(): v for k, v in headers.items()}
    result: Dict[str, Any] = {}
    # Safe keepers
    if "user-agent" in lowered:
        result["user-agent"] = lowered["user-agent"]
    if HEADER_USER_ID in lowered:
        result[HEADER_USER_ID] = lowered[HEADER_USER_ID]
    # Redact auth info
    if "authorization" in lowered:
        auth_val = lowered["authorization"]
        # Keep only the scheme if present
        scheme = auth_val.split(" ")[0] if auth_val else ""
        result["authorization"] = f"{scheme} <redacted>" if scheme else "present"
    return result


def ensure_correlation_id_on_request(request: Any) -> str:
    """Ensure a correlation id exists on the request and return it."""
    if not hasattr(request.ctx, "correlation_id") or not request.ctx.correlation_id:
        request.ctx.correlation_id = uuid.uuid4().hex
    return request.ctx.correlation_id


def extract_request_context() -> Dict[str, Any]:
    """Extract safe request context for logging / Sentry."""
    try:
        request = Request.get_current()
        headers = getattr(request, "headers", {}) or {}
        args = getattr(request, "args", {}) or {}
        json_body = getattr(request, "json", None)
        content_length = getattr(request, "content_length", None)
        ctx: Dict[str, Any] = {
            "method": getattr(request, "method", None),
            "path": getattr(request, "path", None),
            "query_args": dict(args) if hasattr(args, "items") else args,
            "remote_addr": getattr(request, "remote_addr", None),
            "content_length": content_length,
            "has_json": json_body is not None,
            "headers": _sanitize_headers(dict(headers)),
        }
        if hasattr(request, "ctx"):
            ctx["correlation_id"] = ensure_correlation_id_on_request(request)
            # Common custom fields if present
            ctx["user_id"] = request.headers.get(HEADER_USER_ID)
        return ctx
    except Exception:
        return {}


def capture_exception_with_context(
    exc: BaseException,
    event_id: str,
    extra: Optional[Dict[str, Any]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Capture exception in Sentry and log it with rich context.

    Args:
        request: Sanic request
        exc: exception instance
        event_id: structlog event id
        extra: additional context to include
        tags: sentry tags to attach
    """
    request_ctx = extract_request_context()
    if extra is None:
        extra = {}
    # Sentry scope
    try:
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("service", "bot-builder")
            if tags:
                for k, v in tags.items():
                    scope.set_tag(k, v)
            # Flatten some useful fields as tags
            if request_ctx.get("path"):
                scope.set_tag("route", request_ctx["path"])
            if request_ctx.get("method"):
                scope.set_tag("method", request_ctx["method"])
            if request_ctx.get("correlation_id"):
                scope.set_tag("correlation_id", request_ctx["correlation_id"])
            user_id = request_ctx.get("user_id")
            if user_id and hasattr(scope, "set_user"):
                scope.set_user({"id": str(user_id)})
            # Context blocks
            scope.set_context("request", request_ctx)
            if extra:
                scope.set_context("extra", extra)
        sentry_sdk.capture_exception(exc)
    except Exception:
        # Never fail the app because Sentry failed
        pass

    # Structlog error with merged context (avoid dumping huge payloads)
    structlogger.error(
        event_id,
        error=str(exc),
        **{k: v for k, v in {**request_ctx, **extra}.items() if k not in {"headers"}},
    )


def log_request_start(request: Any) -> float:
    """Log request start and return start time."""
    start = time.perf_counter()
    cid = ensure_correlation_id_on_request(request)
    ctx = extract_request_context()
    structlogger.info(
        "request.received",
        method=ctx.get("method"),
        path=ctx.get("path"),
        remote_addr=ctx.get("remote_addr") or "unknown",
        correlation_id=cid,
        user_id=ctx.get("user_id"),
    )
    return start


def log_request_end(request: Any, response: Any, start: float) -> None:
    """Log request completion with latency and correlation id."""
    latency_ms = int((time.perf_counter() - start) * 1000)
    cid = ensure_correlation_id_on_request(request)
    structlogger.info(
        "request.completed",
        method=getattr(request, "method", None),
        path=getattr(request, "path", None),
        status=getattr(response, "status", None),
        latency_ms=latency_ms,
        correlation_id=cid,
    )
