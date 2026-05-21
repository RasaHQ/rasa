"""Event-name constants and small helpers shared across builder Segment modules."""

import datetime as dt
import uuid

# Copilot conversation events
COPILOT_USER_MESSAGE_EVENT = "copilot_user_message"
COPILOT_BOT_MESSAGE_EVENT = "copilot_bot_message"

# MCP tool events
MCP_TOOL_CALLED_EVENT = "mcp_tool_called"


def now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def new_message_id() -> str:
    """Random hex id suitable for a Segment event property."""
    return uuid.uuid4().hex
