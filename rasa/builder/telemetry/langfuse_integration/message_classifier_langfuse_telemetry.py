from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from rasa.builder import config
from rasa.builder.telemetry.langfuse_integration.langfuse_compat import (
    is_langfuse_available,
    langfuse,
)
from rasa.builder.telemetry.langfuse_integration.shared import (
    update_generation_span_with_usage_statistics,
)

if TYPE_CHECKING:
    from rasa.builder.copilot.message_classifier.message_classifier import (
        MessageClassifier,
    )
    from rasa.builder.copilot.message_classifier.models import MessageClassifierResult
    from rasa.builder.copilot.models import CopilotContext


class MessageClassifierLangfuseTelemetry:
    """Telemetry for MessageClassifier LLM classification with Langfuse."""

    @staticmethod
    def trace_classification(
        func: Callable[..., Coroutine[Any, Any, "MessageClassifierResult"]],
    ) -> Callable[..., Coroutine[Any, Any, "MessageClassifierResult"]]:
        """Decorator for tracing MessageClassifier.classify() LLM calls.

        This decorator handles Langfuse tracing for the classification LLM call
        by creating a generation observation within the current trace.

        Args:
            func: The async classify method to wrap.

        Returns:
            Wrapped function with Langfuse telemetry.
        """
        if not is_langfuse_available():
            return func

        @wraps(func)
        async def wrapper(
            self: "MessageClassifier", context: "CopilotContext"
        ) -> "MessageClassifierResult":
            langfuse_client = langfuse.get_client()

            # Extract the last user message for logging
            last_user_message = context.get_last_user_message()
            user_message_text = ""
            if last_user_message:
                # Extract text from content blocks
                for content in last_user_message.content:
                    if hasattr(content, "text"):
                        user_message_text = content.text
                        break

            with langfuse_client.start_as_current_generation(
                name=f"{self.__class__.__name__}.{func.__name__}",
                input={"user_message": user_message_text},
            ) as generation:
                # Call the original classification function
                result = await func(self, context)

                # Import here to avoid circular dependency
                from rasa.builder.copilot.message_classifier.message_classifier import (
                    MessageClassifier,
                )

                # Update the span with classification result and model parameters
                generation.update(
                    model_parameters={
                        "model": config.ORCHESTRATOR_MODEL,
                        "temperature": MessageClassifier.CLASSIFICATION_TEMPERATURE,
                        "max_tokens": MessageClassifier.CLASSIFICATION_MAX_TOKENS,
                    },
                    output={
                        "raw_response": result.raw_response,
                        "category": result.category.value,
                        "requires_full_copilot": result.requires_full_copilot,
                    },
                )

                # Update the span's usage statistics
                if result.classification_usage:
                    update_generation_span_with_usage_statistics(
                        generation, result.classification_usage
                    )

                return result

        return wrapper


class MessageClassifierResponseHandlerLangfuseTelemetry:
    """Telemetry for MessageClassifierResponseHandler LLM generation with Langfuse."""

    @staticmethod
    def trace_response_generation(
        generation_type: str, max_tokens: int
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator factory for tracing greeting/goodbye generation LLM calls.

        This decorator handles Langfuse tracing for response generation
        by creating a generation observation within the current trace.

        Args:
            generation_type: Type of generation ("greeting" or "goodbye").
            max_tokens: Maximum tokens used in the LLM call.

        Returns:
            Decorator function that wraps the streaming method.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if not is_langfuse_available():
                return func

            @wraps(func)
            async def wrapper(self: Any) -> Any:
                langfuse_client = langfuse.get_client()

                with langfuse_client.start_as_current_generation(
                    name=f"MessageClassifierResponseHandler.generate_{generation_type}",
                    input={"user_message": self._user_message},
                ) as generation:
                    output_tokens: list[str] = []

                    # Call the original streaming function and capture output
                    async for token in func(self):
                        if hasattr(token, "content"):
                            output_tokens.append(token.content)
                        yield token

                    # Import here to avoid circular dependency
                    from rasa.builder.copilot.response_handling import (
                        message_classifier_response_handler as handler_module,
                    )

                    # Update the span with model parameters and output
                    handler_class = handler_module.MessageClassifierResponseHandler
                    generation.update(
                        model_parameters={
                            "model": config.ORCHESTRATOR_MODEL,
                            "temperature": handler_class.GENERATION_TEMPERATURE,
                            "max_tokens": max_tokens,
                            "stream": True,
                        },
                        output="".join(output_tokens),
                    )

                    # Update the span's usage statistics
                    if self._generation_usage:
                        update_generation_span_with_usage_statistics(
                            generation, self._generation_usage
                        )

            return wrapper

        return decorator
