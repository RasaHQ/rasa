"""Task that runs the copilot and captures invoked tool calls.

Instantiates ``AgentCopilot`` once and reuses it across items. For each
dataset item, the task drains the response stream, then reads
``response_handler._tracked_tool_calls`` to obtain the ordered list of
tool names invoked. The order reflects invocation order; only events
with ``status == CALLED`` are recorded so each tool call appears once.
"""

import time
from typing import Any, List, Optional

import structlog

from rasa.builder.copilot.models import MCPToolCall, MCPToolCallStatus
from rasa.builder.evaluator.configs.models import ExperimentConfig
from rasa.builder.evaluator.dataset.tool_call_models import ToolCallDatasetEntry
from rasa.builder.evaluator.tasks.base import BaseTask, ToolCallTaskResult

structlogger = structlog.get_logger()


class ToolCallTask(BaseTask):
    """Callable task that runs the copilot on a dataset item and records tool calls."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config)
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        self._copilot = AgentCopilot()

    async def run_task(
        self,
        *,
        item: Any,
        **_: Any,
    ) -> Optional[ToolCallTaskResult]:
        """Run the copilot on a dataset item and return the called-tool list.

        Returns:
            ToolCallTaskResult. On copilot failure, the result contains
            ``error`` set and an empty ``called_tools`` list. Returns None
            only if the dataset item itself cannot be parsed.
        """
        try:
            dataset_entry = ToolCallDatasetEntry.from_raw_data(
                id=item.id,
                input_data=item.input,
                expected_output_data=item.expected_output,
                metadata_data=item.metadata,
            )
        except Exception as e:
            structlogger.error(
                "tasks.tool_call_task.item_parse_failed",
                item_id=getattr(item, "id", None),
                error=str(e),
            )
            return None

        query = dataset_entry.input.query
        context = dataset_entry.to_copilot_context()

        try:
            start = time.monotonic()
            handler, _generation_ctx = await self._copilot.generate_response(context)
            async for _output in handler.stream():
                pass
            latency_ms = (time.monotonic() - start) * 1000

            called_tools = self._extract_called_tools(handler._tracked_tool_calls)

            return ToolCallTaskResult(
                query=query,
                called_tools=called_tools,
                latency_ms=latency_ms,
            )
        except Exception as e:
            structlogger.error(
                "tasks.tool_call_task.copilot_failed",
                item_id=item.id,
                query=query,
                error=str(e),
            )
            return ToolCallTaskResult(
                query=query,
                called_tools=[],
                latency_ms=0.0,
                error=str(e),
            )

    @staticmethod
    def _extract_called_tools(tool_calls: List[MCPToolCall]) -> List[str]:
        """Pick one entry per invocation, preserving order.

        Tool calls flow through the queue as multiple status updates
        (CALLED → RUNNING → COMPLETED/FAILED). We use the CALLED event as
        the canonical "this tool was invoked" marker.
        """
        return [
            tc.tool_name for tc in tool_calls if tc.status == MCPToolCallStatus.CALLED
        ]
