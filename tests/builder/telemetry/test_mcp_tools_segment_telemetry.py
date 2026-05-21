from datetime import datetime
from typing import Dict, List, Tuple

import pytest
from pydantic import BaseModel

from rasa.builder.telemetry.segment_integration.mcp_tools_segment_telemetry import (
    MCPToolsSegmentTelemetry,
)
from rasa.builder.telemetry.segment_integration.shared import MCP_TOOL_CALLED_EVENT


class TrackedEvent(BaseModel):
    event: str
    user_id: str
    properties: Dict


@pytest.fixture
def telemetry_events(
    monkeypatch,
) -> Tuple[MCPToolsSegmentTelemetry, List[TrackedEvent]]:
    """MCP telemetry instance with the underlying `track` patched."""
    events: List[TrackedEvent] = []

    def fake_track(event: str, user_id: str, properties: dict) -> None:
        events.append(TrackedEvent(event=event, user_id=user_id, properties=properties))

    monkeypatch.setattr(
        "rasa.builder.telemetry.segment_integration.mcp_tools_segment_telemetry.track",
        fake_track,
    )
    telemetry = MCPToolsSegmentTelemetry(
        mcp_server="rasa-copilot",
        user_id="user-abc",
    )
    return telemetry, events


def test_init_uses_explicit_user_id_when_provided(monkeypatch):
    """An explicit user_id must be preferred over the resolver fallback."""
    monkeypatch.setattr(
        "rasa.builder.telemetry.segment_integration"
        ".mcp_tools_segment_telemetry.resolve_default_user_id",
        lambda: "should-not-be-used",
    )
    telemetry = MCPToolsSegmentTelemetry(
        mcp_server="rasa-copilot", user_id="explicit-user"
    )
    assert telemetry._user_id == "explicit-user"


def test_init_falls_back_to_resolver_when_user_id_omitted(monkeypatch):
    """When user_id is None, the resolver is consulted."""
    monkeypatch.setattr(
        "rasa.builder.telemetry.segment_integration"
        ".mcp_tools_segment_telemetry.resolve_default_user_id",
        lambda: "resolved-fallback",
    )
    telemetry = MCPToolsSegmentTelemetry(mcp_server="rasa-copilot")
    assert telemetry._user_id == "resolved-fallback"


def test_track_tool_called_success_emits_expected_payload(
    telemetry_events: Tuple[MCPToolsSegmentTelemetry, List[TrackedEvent]],
):
    telemetry, events = telemetry_events
    telemetry.track_tool_called(
        tool_name="search_rasa_docs",
        duration_ms=142,
        success=True,
    )

    assert len(events) == 1
    event = events[0]
    assert event.event == MCP_TOOL_CALLED_EVENT
    assert event.user_id == "user-abc"
    assert event.properties["tool_name"] == "search_rasa_docs"
    assert event.properties["mcp_server"] == "rasa-copilot"
    assert event.properties["duration_ms"] == 142
    assert event.properties["success"] is True
    assert event.properties["error_type"] is None
    assert event.properties["error_message"] is None
    # Sanity-check the timestamp format so a regression in `shared.now_iso`
    # would surface here.
    datetime.fromisoformat(event.properties["timestamp"])


def test_track_tool_called_failure_carries_error_type_and_message(
    telemetry_events: Tuple[MCPToolsSegmentTelemetry, List[TrackedEvent]],
):
    telemetry, events = telemetry_events
    telemetry.track_tool_called(
        tool_name="validate_project",
        duration_ms=37,
        success=False,
        error_type="TimeoutError",
        error_message="server did not respond in time",
    )

    assert len(events) == 1
    event = events[0]
    assert event.properties["success"] is False
    assert event.properties["error_type"] == "TimeoutError"
    assert event.properties["error_message"] == "server did not respond in time"


def test_track_tool_called_error_message_is_optional(
    telemetry_events: Tuple[MCPToolsSegmentTelemetry, List[TrackedEvent]],
):
    telemetry, events = telemetry_events
    telemetry.track_tool_called(
        tool_name="get_flow",
        duration_ms=12,
        success=False,
        error_type="ValueError",
    )

    assert events[0].properties["error_message"] is None
