"""Tests for the MCP tool instrumentation wrapper."""

from typing import Any, Dict, List

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import rasa.builder.telemetry.segment_integration.mcp_tool_instrumentation as module
from rasa.builder.telemetry.segment_integration.mcp_tool_instrumentation import (
    instrument_mcp_tools,
)


class FakeResponse(BaseModel):
    """Response shape used by the test tools; mirrors the real Pydantic models."""

    success: bool = True
    error: str = ""


class _RecordingTelemetry:
    """Stand-in telemetry sender that records ``track_tool_called`` kwargs."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def track_tool_called(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@pytest.fixture
def reset_module_state(monkeypatch):
    """Reset the instrumentation module globals between tests."""
    monkeypatch.setattr(module, "_mcp_telemetry", None)
    monkeypatch.setattr(module, "_mcp_server_name", None)


@pytest.fixture
def telemetry_recorder(reset_module_state, monkeypatch) -> _RecordingTelemetry:
    """Inject a recording telemetry sender into the module."""
    recorder = _RecordingTelemetry()
    monkeypatch.setattr(module, "_mcp_telemetry", recorder)
    return recorder


async def test_instrument_mcp_tools_swaps_tool_method(reset_module_state):
    """The wrapper must replace ``mcp_instance.tool`` so later decorators see it."""
    mcp = FastMCP(name="test")
    original_tool = mcp.tool

    instrument_mcp_tools(mcp, mcp_server="test-server")

    assert mcp.tool is not original_tool


async def test_successful_tool_emits_success_true(
    telemetry_recorder: _RecordingTelemetry,
):
    mcp = FastMCP(name="test")
    instrument_mcp_tools(mcp, mcp_server="test-server")

    @mcp.tool(name="echo")
    async def echo() -> FakeResponse:
        return FakeResponse(success=True)

    await echo()

    assert len(telemetry_recorder.calls) == 1
    call = telemetry_recorder.calls[0]
    assert call["tool_name"] == "echo"
    assert call["success"] is True
    assert call["error_type"] is None
    assert call["error_message"] is None


async def test_reported_failure_response_marks_tool_reported_failure(
    telemetry_recorder: _RecordingTelemetry,
):
    mcp = FastMCP(name="test")
    instrument_mcp_tools(mcp, mcp_server="test-server")

    @mcp.tool(name="bad")
    async def bad() -> FakeResponse:
        return FakeResponse(success=False, error="something broke")

    await bad()

    assert len(telemetry_recorder.calls) == 1
    call = telemetry_recorder.calls[0]
    assert call["tool_name"] == "bad"
    assert call["success"] is False
    assert call["error_type"] == "ToolReportedFailure"
    assert call["error_message"] == "something broke"


async def test_raised_exception_propagates_and_records_exception_class(
    telemetry_recorder: _RecordingTelemetry,
):
    mcp = FastMCP(name="test")
    instrument_mcp_tools(mcp, mcp_server="test-server")

    @mcp.tool(name="boom")
    async def boom() -> FakeResponse:
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await boom()

    assert len(telemetry_recorder.calls) == 1
    call = telemetry_recorder.calls[0]
    assert call["tool_name"] == "boom"
    assert call["success"] is False
    assert call["error_type"] == "RuntimeError"
    assert call["error_message"] == "kaboom"
