from abc import ABC, abstractmethod
from typing import Text, Dict, Any, Iterable
import os
import openai
import logging

logger = logging.getLogger(__name__)


CONFIG_KEY_MODEL_NAME = "sensitive_model_name"
CONFIG_KEY_ACTION = "sensitive_action"
CONFIG_KEY_USE_STUB = "sensitive_use_stub"


class SensitiveTopicDetectorBase(ABC):
    DEFAULT_ACTION = "flow_sensitive-topic"

    def __init__(self, config: Dict[Text, Any]):
        self._action = config.get(CONFIG_KEY_ACTION, self.DEFAULT_ACTION)

    @classmethod
    def get_default_config(cls) -> Dict[Text, Any]:
        return {
            CONFIG_KEY_ACTION: cls.DEFAULT_ACTION,
        }

    @abstractmethod
    def check(self, user_msg: Text) -> bool:
        ...

    def action(self) -> Text:
        return self._action


class SensitiveTopicDetector(SensitiveTopicDetectorBase):
    DEFAULT_MODEL_NAME = "text-davinci-003"
    DEFAULT_USE_STUB = False

    def __init__(self, config: Dict[Text, Any]):
        super().__init__(config)

        # used as a fallback on RateLimit error or OpenAI misconfiguration
        self._stub = SensitiveTopicDetectorStub(config)

        # TODO: move the key in more appropriate config (global DM2 config?)
        key = os.getenv("OPENAI_API_KEY")
        self._model_name = config.get(CONFIG_KEY_MODEL_NAME, self.DEFAULT_MODEL_NAME)
        self._action = config.get(CONFIG_KEY_ACTION, self.DEFAULT_ACTION)
        self._use_stub = config.get(CONFIG_KEY_USE_STUB, self.DEFAULT_USE_STUB)
        if not self._use_stub:
            if key is None:
                logger.warning(f"No OPENAI_API_KEY found in environment, "
                               f"{self.__class__.__name__} uses stub detector")
                self._use_stub = True
            else:
                openai.api_key = key

    @classmethod
    def get_default_config(cls) -> Dict[Text, Any]:
        return {
            CONFIG_KEY_MODEL_NAME: cls.DEFAULT_MODEL_NAME,
            CONFIG_KEY_ACTION: cls.DEFAULT_ACTION,
            CONFIG_KEY_USE_STUB: cls.DEFAULT_USE_STUB,
        }

    def check(self, user_msg: Text) -> bool:
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
            logger.warning("Response: %s -> %s", resp_text.strip(), result)
        except openai.error.RateLimitError:
            logger.warning("RateLimitError from openai, fall back to stub")
            result = self._stub.check(user_msg)
        return result

    def action(self) -> Text:
        return self._action

    @staticmethod
    def _make_prompt(user_msg: Text) -> Text:
        return f"""Below is the message from the user to the specialized
financial chatbot. Can you detect the sensitive topic, not related to
the scope of the bot? Reply "YES" or "NO" and nothing else.

{user_msg}
"""

    @staticmethod
    def _parse_response(text: Text) -> bool:
        return "YES" in text.upper()


class SensitiveTopicDetectorStub(SensitiveTopicDetectorBase):
    """
    Stub class for testing and debugging. Instead of using
    ChatGPT, uses fixed substrings for detection.
    """
    DEFAULT_POSITIVE = (
        "voices in my head",
        "health problems"
    )

    def __init__(self, config: Dict[Text, Any],
                 positive: Iterable[Text] = DEFAULT_POSITIVE):
        super().__init__(config)
        self._positive = list(map(str.lower, positive))

    def check(self, user_msg: Text) -> bool:
        user_msg = user_msg.lower()
        for substr in self._positive:
            if substr in user_msg:
                return True
        return False
