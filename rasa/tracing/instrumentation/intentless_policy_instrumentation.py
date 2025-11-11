import functools
import json
import logging
from typing import TYPE_CHECKING, Callable, List, Optional, Type

from opentelemetry.trace import Tracer

from rasa.shared.core.trackers import DialogueStateTracker

if TYPE_CHECKING:
    from rasa.core.policies.intentless_policy import IntentlessPolicy

logger = logging.getLogger(__name__)


def _instrument_select_response_examples(
    tracer: Tracer, policy_class: Type["IntentlessPolicy"]
) -> None:
    def tracing_select_response_examples_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(
            self: "IntentlessPolicy",
            history: str,
            number_of_samples: int,
            max_number_of_tokens: int,
        ) -> List[str]:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                ai_response_examples = fn(
                    self, history, number_of_samples, max_number_of_tokens
                )
                span.set_attributes(
                    {
                        "ai_response_examples": json.dumps(ai_response_examples),
                    }
                )
                return ai_response_examples

        return wrapper

    policy_class.select_response_examples = tracing_select_response_examples_wrapper(  # type: ignore[method-assign]
        policy_class.select_response_examples
    )

    logger.debug(
        f"Instrumented '{policy_class.__name__}.select_response_examples' method."
    )


def _instrument_select_few_shot_conversations(
    tracer: Tracer, policy_class: Type["IntentlessPolicy"]
) -> None:
    def tracing_select_few_shot_conversations_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(
            self: "IntentlessPolicy",
            history: str,
            number_of_samples: int,
            max_number_of_tokens: int,
        ) -> List[str]:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                conversation_samples = fn(
                    self, history, number_of_samples, max_number_of_tokens
                )
                span.set_attributes(
                    {
                        "conversation_samples": json.dumps(conversation_samples),
                    }
                )
                return conversation_samples

        return wrapper

    policy_class.select_few_shot_conversations = (  # type: ignore[method-assign]
        tracing_select_few_shot_conversations_wrapper(
            policy_class.select_few_shot_conversations
        )
    )

    logger.debug(
        f"Instrumented '{policy_class.__name__}.select_few_shot_conversations' method."
    )


def _instrument_extract_ai_responses(
    tracer: Tracer, policy_class: Type["IntentlessPolicy"]
) -> None:
    def tracing_extract_ai_responses_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(
            self: "IntentlessPolicy", conversation_samples: List[str]
        ) -> List[str]:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                ai_responses = fn(self, conversation_samples)
                span.set_attributes(
                    {
                        "ai_responses": json.dumps(ai_responses),
                    }
                )
                return ai_responses

        return wrapper

    policy_class.extract_ai_responses = tracing_extract_ai_responses_wrapper(  # type: ignore[method-assign]
        policy_class.extract_ai_responses
    )

    logger.debug(f"Instrumented '{policy_class.__name__}.extract_ai_responses' method.")


def _instrument_generate_answer(
    tracer: Tracer, policy_class: Type["IntentlessPolicy"]
) -> None:
    def tracing_generate_answer_wrapper(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(
            self: "IntentlessPolicy",
            response_examples: List[str],
            conversation_samples: List[str],
            history: str,
            tracker: DialogueStateTracker,
        ) -> Optional[str]:
            with tracer.start_as_current_span(
                f"{self.__class__.__name__}.{fn.__name__}"
            ) as span:
                llm_response = await fn(
                    self, response_examples, conversation_samples, history, tracker
                )
                span.set_attributes(
                    {
                        "llm_response": json.dumps(llm_response),
                    }
                )
                return llm_response

        return wrapper

    policy_class.generate_answer = tracing_generate_answer_wrapper(  # type: ignore[method-assign]
        policy_class.generate_answer
    )

    logger.debug(f"Instrumented '{policy_class.__name__}.generate_answer' method.")
