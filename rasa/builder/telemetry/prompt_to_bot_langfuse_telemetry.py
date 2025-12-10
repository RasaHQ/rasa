from typing import TYPE_CHECKING, Any, Dict, Optional

import structlog

from rasa.builder.models import BotFiles
from rasa.builder.telemetry.langfuse_compat import with_langfuse

if TYPE_CHECKING:
    from rasa.builder.copilot.models import UsageStatistics

structlogger = structlog.get_logger()


class PromptToBotLangfuseTelemetry:
    @staticmethod
    def setup_prompt_to_bot_trace(
        prompt: str,
        user_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> None:
        """Set up the Langfuse trace for a prompt-to-bot request.

        Args:
            prompt: The user's prompt for bot generation.
            user_id: Optional user ID for the trace.
            job_id: Optional job ID for the trace.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            # Update the top level trace
            langfuse_client.update_current_trace(
                user_id=user_id,
                session_id=job_id,
                input={"prompt": prompt},
                metadata={
                    "job_id": job_id,
                    "user_id": user_id,
                },
            )
            # Update the current span (run job span)
            langfuse_client.update_current_span(
                input={
                    "prompt": prompt,
                },
                metadata={
                    "job_id": job_id,
                    "user_id": user_id,
                },
            )

    @staticmethod
    def update_prompt_to_bot_trace_output_success(
        bot_files: BotFiles,
        attempts: int,
        max_attempts: int,
    ) -> None:
        """Update the Langfuse trace with the final output.

        Args:
            bot_files: Generated bot files if successful.
            attempts: Number of attempts made.
            max_attempts: Maximum number of attempts allowed.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()

            output: Dict[str, Any] = {
                "success": True,
                "file_count": len(bot_files),
                "files": bot_files,
            }

            metadata: Dict[str, Any] = {
                "max_attempts": max_attempts,
                "attempts": attempts,
            }
            # Update the top level trace
            langfuse_client.update_current_trace(
                output=output,
                metadata=metadata,
            )
            # Update the current span (run job span)
            langfuse_client.update_current_span(
                output=output,
                metadata=metadata,
            )

    @staticmethod
    def update_prompt_to_bot_trace_output_failure(
        error: Exception,
        attempts: int,
        max_attempts: int,
    ) -> None:
        """Update the Langfuse trace with the final output.

        Args:
            error: The exception that caused the failure.
            attempts: Number of attempts made.
            max_attempts: Maximum number of attempts allowed.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()

            output: Dict[str, Any] = {
                "success": False,
                "error": str(error),
            }

            metadata: Dict[str, Any] = {
                "max_attempts": max_attempts,
                "attempts": attempts,
            }

            langfuse_client.update_current_trace(
                output=output,
                metadata=metadata,
            )

    @staticmethod
    def update_current_span_with_usage_statistics(
        usage_statistics: "UsageStatistics",
    ) -> None:
        """Update the current Langfuse span with usage statistics.

        Args:
            usage_statistics: The usage statistics to add to the span.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            # Update the span with usage statistics in Langfuse format
            langfuse_client.update_current_generation(
                usage_details={
                    "input_non_cached_usage": (
                        usage_statistics.non_cached_prompt_tokens or 0
                    ),
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
