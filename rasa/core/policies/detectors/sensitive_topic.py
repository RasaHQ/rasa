from abc import ABC, abstractmethod
import importlib.resources
from typing import Optional, Text, Dict, Any, Iterable
import os
import openai
import openai.error
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)


CONFIG_KEY_MODEL_NAME = "sensitive_model_name"
CONFIG_KEY_ACTION = "sensitive_action"
CONFIG_KEY_USE_STUB = "sensitive_use_stub"
CONFIG_KEY_PROMPT_TEMPLATE = "prompt"

DEFAULT_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.core.policies.detectors", "prompt_sensitive_topic.jinja2"
)


class SensitiveTopicDetectorBase(ABC):
    """Base class for sensitive topic detectors."""

    DEFAULT_ACTION = "flow_sensitive-topic"

    def __init__(self, config: Dict[Text, Any]):
        self._action = config.get(CONFIG_KEY_ACTION, self.DEFAULT_ACTION)

    @classmethod
    def get_default_config(cls) -> Dict[Text, Any]:
        return {
            # action to be executed if the sensitive topic is detected
            CONFIG_KEY_ACTION: cls.DEFAULT_ACTION,
        }

    @abstractmethod
    def check(self, user_msg: Text) -> bool:
        """Check if the user message contains sensitive topic.

        Args:
            user_msg: user message to check

        Returns:
            True if the message contains sensitive topic, False otherwise"""
        ...

    def action(self) -> Text:
        """Return action to be executed if the sensitive topic is detected."""
        return self._action


class SensitiveTopicDetector(SensitiveTopicDetectorBase):
    """Sensitive topic detector based on OpenAI ChatGPT model."""

    # TODO: move to shared configuration of components based on LLM
    DEFAULT_MODEL_NAME = "text-davinci-003"
    DEFAULT_USE_STUB = False

    def __init__(self, config: Dict[Text, Any]):
        super().__init__(config)

        # used as a fallback on RateLimit error or OpenAI misconfiguration
        self._stub = SensitiveTopicDetectorStub(config)

        # TODO: move the key in more appropriate config (global detector config?)
        key = os.getenv("OPENAI_API_KEY")

        self._model_name = config.get(CONFIG_KEY_MODEL_NAME, self.DEFAULT_MODEL_NAME)
        self._action = config.get(CONFIG_KEY_ACTION, self.DEFAULT_ACTION)
        self._use_stub = config.get(CONFIG_KEY_USE_STUB, self.DEFAULT_USE_STUB)
        self._prompt_template = config.get(
            CONFIG_KEY_PROMPT_TEMPLATE, DEFAULT_PROMPT_TEMPLATE
        )

        if not self._use_stub:
            if key is None:
                logger.warning(
                    f"No OPENAI_API_KEY found in environment, "
                    f"{self.__class__.__name__} uses stub detector"
                )
                self._use_stub = True
            else:
                openai.api_key = key

    @classmethod
    def get_default_config(cls) -> Dict[Text, Any]:
        default_config = super().get_default_config()
        default_config.update(
            {
                CONFIG_KEY_MODEL_NAME: cls.DEFAULT_MODEL_NAME,
                CONFIG_KEY_USE_STUB: cls.DEFAULT_USE_STUB,
                CONFIG_KEY_PROMPT_TEMPLATE: DEFAULT_PROMPT_TEMPLATE,
            }
        )

        return default_config

    def check(self, user_msg: Optional[Text]) -> bool:
        if not user_msg:
            return False

        if self._use_stub:
            return self._stub.check(user_msg)
        try:
            resp = openai.Completion.create(
                model=self._model_name,
                prompt=self._make_prompt(user_msg),
                temperature=0.0,
            )
            resp_text = resp.choices[0].text
            result = self._parse_response(resp_text)
            logger.info("Response: %s -> %s", resp_text.strip(), result)
        except openai.error.RateLimitError:
            logger.warning("RateLimitError from openai, fall back to stub")
            result = self._stub.check(user_msg)
        return result

    def _make_prompt(self, user_message: Text) -> Text:
        """Make prompt for OpenAI ChatGPT model."""
        return Template(self._prompt_template).render(user_message=user_message)

    @staticmethod
    def _parse_response(text: Text) -> bool:
        """Parse response from OpenAI ChatGPT model.

        Expected responses are "YES" and "NO" (case-insensitive)."""
        return "YES" in text.upper()


class SensitiveTopicDetectorStub(SensitiveTopicDetectorBase):
    """Stub class for testing and debugging.

    Instead of using ChatGPT, uses fixed substrings for detection.
    """

    DEFAULT_POSITIVE = ("voices in my head", "health problems")

    def __init__(
        self, config: Dict[Text, Any], positive: Iterable[Text] = DEFAULT_POSITIVE
    ):
        super().__init__(config)
        self._positive = list(map(str.lower, positive))

    def check(self, user_msg: Text) -> bool:
        user_msg = user_msg.lower()
        for substr in self._positive:
            if substr in user_msg:
                return True
        return False
