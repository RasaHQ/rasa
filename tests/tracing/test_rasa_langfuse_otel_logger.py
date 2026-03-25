"""Tests for Rasa Langfuse OTEL logger (assistant text + tool calls)."""

from rasa.tracing.rasa_langfuse_otel_logger import (
    maybe_merged_assistant_content_and_tool_calls,
)


def test_merge_when_content_and_tool_calls_present() -> None:
    response_obj = {
        "id": "resp-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Checking that for you.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "x"}',
                            },
                        }
                    ],
                }
            }
        ],
    }
    merged = maybe_merged_assistant_content_and_tool_calls(response_obj)
    assert merged is not None
    assert merged["role"] == "assistant"
    assert merged["content"] == "Checking that for you."
    assert len(merged["tool_calls"]) == 1
    assert merged["tool_calls"][0]["name"] == "search"
    assert merged["tool_calls"][0]["call_id"] == "call_abc"
    assert merged["tool_calls"][0]["arguments"] == {"q": "x"}


def test_no_merge_when_tool_calls_only() -> None:
    response_obj = {
        "id": "resp-2",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "function": {"name": "done", "arguments": "{}"},
                        }
                    ],
                }
            }
        ],
    }
    assert maybe_merged_assistant_content_and_tool_calls(response_obj) is None


def test_no_merge_when_content_only() -> None:
    response_obj = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                }
            }
        ],
    }
    assert maybe_merged_assistant_content_and_tool_calls(response_obj) is None
