from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from rasa.builder import config
from rasa.builder.copilot.models import UsageStatistics
from rasa.builder.telemetry.langfuse.langfuse_compat import (
    is_langfuse_available,
    langfuse,
)
from rasa.builder.telemetry.langfuse.shared import (
    update_generation_span_with_usage_statistics,
)

if TYPE_CHECKING:
    from rasa.builder.document_retrieval.inkeep_document_retrieval import (
        InKeepDocumentRetrieval,
    )


class DocumentRetrievalLangfuseTelemetry:
    """Telemetry for document retrieval generation with Langfuse."""

    @staticmethod
    def trace_document_retrieval_generation(
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
            self: "InKeepDocumentRetrieval",
            query: str,
            temperature: float,
            timeout: float,
        ) -> Any:
            langfuse_client = langfuse.get_client()

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={
                    "query": query,
                    "temperature": temperature,
                    "timeout": timeout,
                },
            ) as generation:
                # Call the original function
                response = await func(self, query, temperature, timeout)

                # Update the span with response content
                generation.update(
                    output=response,
                    model_parameters={
                        "temperature": str(temperature),
                        "timeout": str(timeout),
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
