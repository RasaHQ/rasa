"""Startup validation for builder configuration values.

Call ``validate_config()`` early in the application lifecycle (e.g. from
``main()``) to catch misconfigurations before they surface as confusing
runtime errors.
"""

import structlog

from rasa.builder import config

structlogger = structlog.get_logger()


def validate_config() -> None:
    """Validate cross-field configuration invariants.

    Raises:
        ValueError: If any invariant is violated.
    """
    _validate_stream_timeout_exceeds_mcp_timeout()


def _validate_stream_timeout_exceeds_mcp_timeout() -> None:
    """COPILOT_MAX_NEXT_STREAM_EVENT_WAIT_TIME_SECONDS > MCP_TOOL_CALL_TIMEOUT.

    Slow MCP tool calls must always time out via the MCP layer first,
    giving the correct failure mode instead of a premature
    CopilotNextStreamEventTimeoutException.
    """
    stream_timeout = config.COPILOT_MAX_NEXT_STREAM_EVENT_WAIT_TIME_SECONDS
    mcp_timeout = config.MCP_TOOL_CALL_TIMEOUT
    if stream_timeout <= mcp_timeout:
        structlogger.error(
            "builder.config_validation.stream_timeout_exceeds_mcp_timeout",
            event_info="Stream timeout must be greater than MCP timeout",
            stream_timeout=stream_timeout,
            mcp_timeout=mcp_timeout,
        )
        raise ValueError(
            f"COPILOT_MAX_NEXT_STREAM_EVENT_WAIT_TIME_SECONDS "
            f"({config.COPILOT_MAX_NEXT_STREAM_EVENT_WAIT_TIME_SECONDS}) must be "
            f"strictly greater than MCP_TOOL_CALL_TIMEOUT "
            f"({config.MCP_TOOL_CALL_TIMEOUT})."
        )
