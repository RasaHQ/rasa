from functools import wraps
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List

from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
    is_langfuse_available,
    langfuse,
)
from rasa.builder.telemetry.langfuse_integration.shared import (
    update_generation_span_with_usage_statistics,
)

if TYPE_CHECKING:
    from rasa.builder.copilot.legacy_copilot import LegacyCopilot


class LegacyCopilotLangfuseTelemetry:
    """Telemetry for legacy copilot LLM streaming generation with Langfuse."""

    @staticmethod
    def trace_streaming_generation(
        func: Callable[..., AsyncGenerator[str, None]],
    ) -> Callable[..., AsyncGenerator[str, None]]:
        """Custom decorator for tracing async streaming of the Copilot's LLM generation.

        This decorator handles Langfuse tracing for async streaming of the Legacy
        Copilot's LLM generation by manually managing the generation span and updating
        it with usage statistics after the stream completes.
        """
        if not is_langfuse_available():
            return func

        @wraps(func)
        async def wrapper(
            self: "LegacyCopilot", messages: List[Dict[str, Any]]
        ) -> AsyncGenerator[str, None]:
            langfuse_client = langfuse.get_client()

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={"messages": messages},
            ) as generation:
                output: list[str] = []
                # Call the original streaming function and start capturing the output
                async for chunk in func(self, messages):
                    output.append(chunk)
                    yield chunk

                # Update the span's model parameters and output after streaming is
                # complete
                generation.update(
                    model_parameters=self.llm_config, output="".join(output)
                )

                # Update the span's usage statistics after streaming is complete
                if self.usage_statistics:
                    update_generation_span_with_usage_statistics(
                        generation, self.usage_statistics
                    )

        return wrapper
