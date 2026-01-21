"""MessageClassifier - Classifies user requests for routing.

The message classifier uses a lightweight LLM (gpt-4o-mini) to classify user requests
and decide whether to handle them directly or delegate to the full copilot.

Benefits:
- Dedicated logic for more precise categorization
- Simple requests for faster responses and lower cost
- Clean separation between classification and generation
"""

import importlib.resources
import re
from typing import ClassVar, Dict, List

import structlog
from jinja2 import Template
from openai import AsyncOpenAI

from rasa.builder import config
from rasa.builder.copilot.constants import (
    MESSAGE_CLASSIFIER_PROMPT_FILE,
    MESSAGE_CLASSIFIER_PROMPTS_DIR,
)
from rasa.builder.copilot.message_classifier.models import MessageClassifierResult
from rasa.builder.copilot.models import ResponseCategory, UsageStatistics
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()

# Token pattern for extraction from LLM response
TOKEN_PATTERN = re.compile(r"\[([A-Z_]+)\]")


class MessageClassifier:
    """Classifies user requests and routes them to the appropriate handler.

    Uses a lightweight LLM (e.g. gpt-4o-mini) to categorize incoming requests.
    Technical questions are routed to the full copilot agent, while simple
    requests (greetings, goodbyes) and out-of-scope queries are handled directly.

    The focused classification model provides more accurate routing decisions
    than embedding classification logic in the main copilot prompt.
    """

    # Categories that the MessageClassifier can classify into
    CLASSIFIER_CATEGORIES: ClassVar[List[ResponseCategory]] = [
        ResponseCategory.GREETING_DETECTION,
        ResponseCategory.GOODBYE_DETECTION,
        ResponseCategory.ROLEPLAY_DETECTION,
        ResponseCategory.OUT_OF_SCOPE_DETECTION,
        ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
        ResponseCategory.UNCLEAR_INPUT_DETECTION,
        ResponseCategory.COPILOT,
    ]

    # Map classifier tokens to ResponseCategory
    TOKEN_TO_CATEGORY: ClassVar[Dict[str, ResponseCategory]] = {
        f"[{category.value.upper()}]": category for category in CLASSIFIER_CATEGORIES
    }

    def __init__(self) -> None:
        self._client = AsyncOpenAI()
        self._prompt_template = self._load_prompt_template()

    @staticmethod
    def _load_prompt_template() -> Template:
        """Load the orchestrator prompt template from resources.

        Returns:
            Template object containing the prompt template.
        """
        template_content = importlib.resources.read_text(
            f"{PACKAGE_NAME}.{MESSAGE_CLASSIFIER_PROMPTS_DIR}",
            MESSAGE_CLASSIFIER_PROMPT_FILE,
        )
        return Template(template_content)

    async def classify(self, user_message: str) -> MessageClassifierResult:
        """Classify a user message and decide how to handle it.

        Args:
            user_message: The user's message to classify.

        Returns:
            MessageClassifierResult with the decision category and usage stats.
        """
        prompt = self._prompt_template.render(user_message=user_message)

        structlogger.debug(
            "classifier.classify.start",
            event_info="Starting orchestrator classification",
            message_preview=user_message[:100],
        )

        try:
            response = await self._client.chat.completions.create(
                model=config.ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                ],
                max_tokens=50,  # Classifications are short
                temperature=0,  # Deterministic classification
            )

            raw_response = (response.choices[0].message.content or "").strip()
            category = self._parse_category(raw_response)

            # Create usage statistics for classification
            classification_usage = UsageStatistics(
                model=config.ORCHESTRATOR_MODEL,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else 0
                ),
                total_tokens=response.usage.total_tokens if response.usage else 0,
                cached_prompt_tokens=0,
                input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
            )

            structlogger.info(
                "classifier.classify.success",
                event_info="Orchestrator classification completed",
                category=category.value,
                raw_response=raw_response,
            )

            return MessageClassifierResult(
                category=category,
                classification_usage=classification_usage,
            )

        except Exception as e:
            structlogger.error(
                "classifier.classify.error",
                event_info="Orchestrator classification failed",
                error=str(e),
            )
            # Default to triggering copilot on error (fail-safe)
            return MessageClassifierResult(
                category=ResponseCategory.COPILOT,
                classification_usage=UsageStatistics(
                    model=config.ORCHESTRATOR_MODEL,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cached_prompt_tokens=0,
                    input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                    output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                    cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
                ),
            )

    def _parse_category(self, raw_response: str) -> ResponseCategory:
        """Parse the LLM response into a ResponseCategory.

        Args:
            raw_response: Raw text response from the classification LLM.

        Returns:
            ResponseCategory enum value.
        """
        # Extract token from response using regex
        if match := TOKEN_PATTERN.search(raw_response):
            token = f"[{match.group(1)}]"
            if token in self.TOKEN_TO_CATEGORY:
                return self.TOKEN_TO_CATEGORY[token]

        # If we can't parse, default to copilot (fail-safe)
        structlogger.warning(
            "classifier.parse.fallback",
            event_info="Could not parse orchestrator response, defaulting to COPILOT",
            raw_response=raw_response,
        )
        return ResponseCategory.COPILOT


# Singleton orchestrator instance (lazy initialization)
