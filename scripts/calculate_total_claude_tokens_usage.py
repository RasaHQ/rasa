#!/usr/bin/env python
"""Total Claude tokens usage.

Calculate the aggregate Claude tokens usage metrics,
on `analyse-and-fix-ci-failures.yml` workflow run.
"""


import json
import os
import sys
from dataclasses import dataclass
from typing import Any

CLAUDE_OUTPUT_FILE_PATH = os.getenv("CLAUDE_OUTPUT_FILE_PATH",
                                    "claude-output.jsonl")


@dataclass(kw_only=True)
class CacheCreation:
    """Class representing `usage.cache_creation` item from Claude output."""
    ephemeral_5m_input_tokens: float = 0
    ephemeral_1h_input_tokens: float = 0


@dataclass(kw_only=True)
class ServerToolUse:
    """Class representing `usage.server_tool_use` item from Claude output."""
    web_search_requests: float = 0


class Usage:
    """Class representing `usage` item from Claude output data."""
    def __init__(self, cache_creation: dict[str, float] | None = None,
                 server_tool_use: dict[str, float] | None = None,
                 input_tokens: float = 0,
                 cache_creation_input_tokens: float = 0,
                 cache_read_input_tokens: float = 0, output_tokens: float = 0,
                 service_tier: str = "standard"):
        self.input_tokens = input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation = CacheCreation(**cache_creation if isinstance(cache_creation, dict) else {})
        self.server_tool_use = ServerToolUse(**server_tool_use if isinstance(server_tool_use, dict) else {})
        self.output_tokens = output_tokens
        self.service_tier = service_tier


def _get_structured_claude_output_data(
        claude_output_file_path: str) -> list[dict[str, Any]]:
    claude_outputs: list[dict[str, Any]] = []
    # Parse Claude output from JSONL file.
    try:
        with open(claude_output_file_path, "r", encoding="utf-8") as data:
            for line in data:
                if valid_line := line.strip():
                    try:
                        line_to_be_added = json.loads(valid_line)
                        if isinstance(line_to_be_added, dict):
                            claude_outputs.append(line_to_be_added)
                    # Skip if a line is not valid JSON
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        message = f"Claude output file '{claude_output_file_path}' not found"
        sys.exit(message)

    if claude_outputs:
        return claude_outputs
    sys.exit(f"'{claude_output_file_path}' does not contain valid JSON")


def aggregate_token_usage_metrics(
        claude_output_data: list[dict[str, Any]]) -> Usage:
    """Aggregate token usage metrics, from claude multi-turn/iterations data.

    Args:
        claude_output_data: Structured Claude results output

    Returns:
        Usage: Aggregated metrics from multi-turn/iteration data
    """
    usages: list[Usage] = [Usage()]

    # Get list of usages
    for data in claude_output_data:
        for k, v in data.items():
            if k == "usage":
                usages.append(Usage(**v) if isinstance(v, dict) else Usage())
            elif k == "message" and isinstance(v, dict) and "usage" in v:
                usages.append(Usage(**v["usage"]) if isinstance(v["usage"],
                                                                dict) else Usage())

    # Aggregate the usages
    aggregated_usage = Usage()
    for usage in usages:
        aggregated_usage.input_tokens += usage.input_tokens
        aggregated_usage.cache_creation_input_tokens += usage.cache_creation_input_tokens
        aggregated_usage.cache_read_input_tokens += usage.cache_read_input_tokens
        aggregated_usage.output_tokens += usage.output_tokens
        aggregated_usage.cache_creation.ephemeral_1h_input_tokens += usage.cache_creation.ephemeral_1h_input_tokens
        aggregated_usage.cache_creation.ephemeral_5m_input_tokens += usage.cache_creation.ephemeral_5m_input_tokens
        aggregated_usage.server_tool_use.web_search_requests += usage.server_tool_use.web_search_requests

    return aggregated_usage


if __name__ == "__main__":
    claude_data = _get_structured_claude_output_data(CLAUDE_OUTPUT_FILE_PATH)
    total_usage = aggregate_token_usage_metrics(claude_data)
    print(json.dumps({
        "total_input_tokens": total_usage.input_tokens,
        "total_cache_creation_input_tokens": total_usage.cache_creation_input_tokens,
        "total_cache_read_input_tokens": total_usage.cache_read_input_tokens,
        "total_ephemeral_1h_input_tokens": total_usage.cache_creation.ephemeral_1h_input_tokens,
        "total_ephemeral_5m_input_tokens": total_usage.cache_creation.ephemeral_5m_input_tokens,
        "total_output_tokens": total_usage.output_tokens,
        "total_websearch_requests": total_usage.server_tool_use.web_search_requests,
        "service_tier": total_usage.service_tier
    }))
