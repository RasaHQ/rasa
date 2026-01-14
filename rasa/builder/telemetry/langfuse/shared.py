from typing import Any

from rasa.builder.copilot.models import UsageStatistics


def update_generation_span_with_usage_statistics(
    generation_span: Any,
    usage_statistics: UsageStatistics,
) -> None:
    """Update the generation span with the usage statistics.

    Args:ga
        generation_span: The generation span.
        usage_statistics: The usage statistics of the generation.
    """
    generation_span.update(
        usage_details={
            "input_non_cached_usage": (usage_statistics.non_cached_prompt_tokens or 0),
            "input_cached_usage": usage_statistics.cached_prompt_tokens or 0,
            "output_usage": usage_statistics.completion_tokens or 0,
            "total": usage_statistics.total_tokens or 0,
        },
        cost_details={
            "input_non_cached_cost": usage_statistics.non_cached_cost or 0,
            "input_cached_cost": usage_statistics.cached_cost or 0,
            "output_cost": usage_statistics.output_cost or 0,
            "total": usage_statistics.total_cost or 0,
        },
        model=usage_statistics.model,
    )
