"""Rasa-owned LiteLLM Langfuse OTEL logger with correct mixed assistant output.

Upstream ``LangfuseOtelLogger._set_observation_output`` writes only tool calls to
``langfuse.observation.output`` when ``tool_calls`` is present, dropping
``message.content``. MCP and other agents often stream text and tools together.

This module subclasses ``LangfuseOtelLogger`` and registers an instance on
``litellm.success_callback`` instead of the string ``\"langfuse_otel\"``, so
LiteLLM invokes our ``set_attributes`` override. We delegate attribute wiring to
upstream, then overwrite ``langfuse.observation.output`` when both assistant
text and tool calls are present.

Long term, the same logic should live in LiteLLM (see BerriAI/litellm); after
an upstream release contains the fix, this subclass can be removed and Rasa can
revert to ``success_callback = [\"langfuse_otel\"]``.
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
from litellm.types.integrations.langfuse_otel import LangfuseSpanAttributes


def maybe_merged_assistant_content_and_tool_calls(
    response_obj: Any,
) -> Optional[dict[str, Any]]:
    """If the first choice has both content and tool calls, return merged dict."""
    if not response_obj or not hasattr(response_obj, "get"):
        return None

    choices = response_obj.get("choices", [])
    if not choices:
        return None

    first_choice = choices[0]
    if first_choice is None or not hasattr(first_choice, "get"):
        return None

    message = first_choice.get("message")
    if message is None or not hasattr(message, "get"):
        return None

    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return None

    content = message.get("content")
    if content is None or content == "":
        return None

    transformed = _transform_tool_calls_for_langfuse(response_obj, tool_calls)
    return {
        "role": message.get("role", "assistant"),
        "content": content,
        "tool_calls": transformed,
    }


def _transform_tool_calls_for_langfuse(
    response_obj: Any, tool_calls: List[Any]
) -> List[dict[str, Any]]:
    """Match LiteLLM's Langfuse OTEL tool-call shape."""
    out: List[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {}) if hasattr(tool_call, "get") else {}
        arguments_str = (
            function.get("arguments", "{}") if hasattr(function, "get") else "{}"
        )
        try:
            arguments_obj = (
                json.loads(arguments_str)
                if isinstance(arguments_str, str)
                else arguments_str
            )
        except json.JSONDecodeError:
            arguments_obj = {}
        call_id = tool_call.get("id", "") if hasattr(tool_call, "get") else ""
        name = function.get("name", "") if hasattr(function, "get") else ""
        out.append(
            {
                "id": response_obj.get("id", ""),
                "name": name,
                "call_id": call_id,
                "type": "function_call",
                "arguments": arguments_obj,
            }
        )
    return out


def _observation_output_json(merged: dict[str, Any]) -> str:
    """Serialize merged assistant payload for OTEL (upstream safe_dumps role)."""
    return json.dumps(merged, default=str)


class RasaLangfuseOtelLogger(LangfuseOtelLogger):
    """Langfuse OTEL logger that preserves assistant text alongside tool calls."""

    def __init__(self, config: Any = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)  # type: ignore[no-untyped-call]

    def set_attributes(
        self, span: Any, kwargs: Any, response_obj: Optional[Any]
    ) -> None:
        """Delegate to LiteLLM, then fix ``langfuse.observation.output`` when needed.

        LiteLLM's ``OpenTelemetry.set_attributes`` hardcodes
        ``LangfuseOtelLogger.set_langfuse_otel_attributes`` for the string
        callback ``langfuse_otel``. Using a registered instance avoids that
        indirection so this override is actually invoked.
        """
        LangfuseOtelLogger.set_langfuse_otel_attributes(span, kwargs, response_obj)
        merged = maybe_merged_assistant_content_and_tool_calls(response_obj)
        if merged is not None:
            span.set_attribute(
                LangfuseSpanAttributes.OBSERVATION_OUTPUT.value,
                _observation_output_json(merged),
            )


def build_rasa_langfuse_otel_logger() -> RasaLangfuseOtelLogger:
    """Construct the logger LiteLLM should use for Langfuse OTEL export."""
    return RasaLangfuseOtelLogger(config=None, callback_name="langfuse_otel")
