from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from rasa.builder import config
from rasa.builder.copilot.models import UsageStatistics
from rasa.builder.telemetry.langfuse_compat import (
    is_langfuse_available,
    langfuse,
    with_langfuse,
)
from rasa.builder.telemetry.shared import update_generation_span_with_usage_statistics

if TYPE_CHECKING:
    from rasa.builder.project_generator.project_generator import ProjectGenerator


class WelcomeMessageGenerationLangfuseTelemetry:
    """Telemetry utilities for welcome message generation traces."""

    @staticmethod
    def update_welcome_message_generation_input(
        flows: str,
        prompt: str,
    ) -> None:
        """Update the current Langfuse span with welcome message generation input.

        Args:
            flows: The generated flows data.
            prompt: The full prompt sent to the LLM.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            langfuse_client.update_current_span(
                input={
                    "flows": flows,
                    "prompt": prompt,
                }
            )

    @staticmethod
    def update_welcome_message_generation_output(
        response_content: str,
        welcome_message: str,
    ) -> None:
        """Update the current Langfuse span with welcome message generation output.

        Args:
            response_content: The response content from the LLM.
            example_questions: The extracted example questions from the response.
            welcome_message: The cleaned and validated welcome message.
        """
        with with_langfuse() as lf:
            if not lf:
                return
            langfuse_client = lf.get_client()
            langfuse_client.update_current_span(
                output={
                    "response_content": response_content,
                    "welcome_message": welcome_message,
                }
            )

    @staticmethod
    def trace_text_generation(
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Custom decorator for tracing document retrieval generation with Langfuse.

        This decorator handles Langfuse tracing for document retrieval API calls
        by manually managing the generation span and updating it with usage statistics.
        """
        if not is_langfuse_available():
            return func

        @wraps(func)
        async def wrapper(
            self: "ProjectGenerator",
            prompt: str,
            max_completion_tokens: float,
        ) -> Any:
            langfuse_client = langfuse.get_client()

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={
                    "prompt": prompt,
                    "max_completion_tokens": max_completion_tokens,
                },
            ) as generation:
                # Call the original function
                response = await func(self, prompt, max_completion_tokens)
                # Update the span with response content
                generation.update(
                    output=response,
                    model_parameters={
                        "max_completion_tokens": str(max_completion_tokens),
                    },
                )

                # Update usage statistics if available
                usage_statistics = UsageStatistics.from_chat_completion_response(
                    response,
                    input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                    output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                    cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
                )

                if usage_statistics:
                    update_generation_span_with_usage_statistics(
                        generation, usage_statistics
                    )

                return response

        return wrapper
