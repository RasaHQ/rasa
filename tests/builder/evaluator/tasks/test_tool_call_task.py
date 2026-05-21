"""Tests for ToolCallTask."""

from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

from rasa.builder.copilot.models import MCPToolCall, MCPToolCallStatus
from rasa.builder.evaluator.configs.models import ExperimentConfig
from rasa.builder.evaluator.tasks.base import ToolCallTaskResult
from rasa.builder.evaluator.tasks.tool_call_task import ToolCallTask

AGENT_COPILOT_PATH = "rasa.builder.copilot.agent_sdk.agent_copilot.AgentCopilot"


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="test",
        description="test",
        dataset_name="test",
        task="tool_call",
        results_dir="/tmp/test",
        formats=["yaml"],
    )


def _make_item(query: str = "list all flows") -> SimpleNamespace:
    return SimpleNamespace(
        id="item-1",
        input={"query": query},
        expected_output={"expected_tools": ["list_flows"]},
        metadata={"category": "flow-edit", "source": "manual"},
    )


def _make_tool_call(name: str, status: MCPToolCallStatus) -> MCPToolCall:
    return MCPToolCall(tool_name=name, status=status)


def _make_handler(tool_calls: List[MCPToolCall], stream_raises: bool = False):
    """Build a mock response handler with a drainable stream."""
    handler = MagicMock()
    handler._tracked_tool_calls = tool_calls

    if stream_raises:

        async def _stream():
            raise RuntimeError("stream blew up")
            yield  # pragma: no cover — make this an async generator

        handler.stream = _stream
    else:

        async def _stream():
            for tc in tool_calls:
                yield tc

        handler.stream = _stream

    return handler


def _patch_copilot(handler=None, generate_raises: bool = False):
    """Patch AgentCopilot so its instance returns the given handler."""
    instance = MagicMock()
    if generate_raises:
        instance.generate_response = AsyncMock(
            side_effect=RuntimeError("generate blew up")
        )
    else:
        instance.generate_response = AsyncMock(return_value=(handler, MagicMock()))
    return patch(AGENT_COPILOT_PATH, return_value=instance), instance


class TestToolCallTaskInit:
    def test_constructs_agent_copilot_once(self):
        with patch(AGENT_COPILOT_PATH) as mock_cls:
            task = ToolCallTask(config=_make_config())
            mock_cls.assert_called_once_with()
            assert task._copilot is mock_cls.return_value


class TestToolCallTaskRunTask:
    async def test_happy_path_records_called_events_in_order(self):
        # Two tools, each emitting CALLED → RUNNING → COMPLETED. Only the
        # CALLED events should appear in called_tools, in the order they
        # were emitted.
        tool_calls = [
            _make_tool_call("list_flows", MCPToolCallStatus.CALLED),
            _make_tool_call("list_flows", MCPToolCallStatus.RUNNING),
            _make_tool_call("list_flows", MCPToolCallStatus.COMPLETED),
            _make_tool_call("read_file", MCPToolCallStatus.CALLED),
            _make_tool_call("read_file", MCPToolCallStatus.RUNNING),
            _make_tool_call("read_file", MCPToolCallStatus.COMPLETED),
        ]
        handler = _make_handler(tool_calls)
        patcher, _ = _patch_copilot(handler=handler)

        with patcher:
            task = ToolCallTask(config=_make_config())
            result = await task.run_task(item=_make_item())

        assert isinstance(result, ToolCallTaskResult)
        assert result.query == "list all flows"
        assert result.called_tools == ["list_flows", "read_file"]
        assert result.latency_ms > 0
        assert result.error is None

    async def test_preserves_repeat_invocations_in_order(self):
        # Same tool called twice — both CALLED events should appear.
        tool_calls = [
            _make_tool_call("a", MCPToolCallStatus.CALLED),
            _make_tool_call("b", MCPToolCallStatus.CALLED),
            _make_tool_call("a", MCPToolCallStatus.CALLED),
        ]
        handler = _make_handler(tool_calls)
        patcher, _ = _patch_copilot(handler=handler)

        with patcher:
            task = ToolCallTask(config=_make_config())
            result = await task.run_task(item=_make_item())

        assert result.called_tools == ["a", "b", "a"]

    async def test_filters_out_non_called_statuses(self):
        # Only RUNNING/COMPLETED/FAILED — no CALLED events at all.
        tool_calls = [
            _make_tool_call("a", MCPToolCallStatus.RUNNING),
            _make_tool_call("a", MCPToolCallStatus.COMPLETED),
            _make_tool_call("b", MCPToolCallStatus.FAILED),
        ]
        handler = _make_handler(tool_calls)
        patcher, _ = _patch_copilot(handler=handler)

        with patcher:
            task = ToolCallTask(config=_make_config())
            result = await task.run_task(item=_make_item())

        assert result.called_tools == []

    async def test_generate_response_failure_returns_error_result(self):
        patcher, _ = _patch_copilot(generate_raises=True)

        with patcher:
            task = ToolCallTask(config=_make_config())
            result = await task.run_task(item=_make_item())

        assert isinstance(result, ToolCallTaskResult)
        assert result.error == "generate blew up"
        assert result.called_tools == []
        assert result.latency_ms == 0.0
        assert result.query == "list all flows"

    async def test_stream_failure_returns_error_result(self):
        handler = _make_handler([], stream_raises=True)
        patcher, _ = _patch_copilot(handler=handler)

        with patcher:
            task = ToolCallTask(config=_make_config())
            result = await task.run_task(item=_make_item())

        assert isinstance(result, ToolCallTaskResult)
        assert result.error == "stream blew up"
        assert result.called_tools == []
        assert result.query == "list all flows"

    async def test_item_parse_failure_returns_none(self):
        with patch(AGENT_COPILOT_PATH):
            task = ToolCallTask(config=_make_config())

        bad_item = SimpleNamespace(
            id="bad-item",
            input={},  # missing required `query`
            expected_output={"expected_tools": []},
            metadata={"category": "x", "source": "y"},
        )
        result = await task.run_task(item=bad_item)
        assert result is None
