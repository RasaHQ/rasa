from abc import ABC, abstractmethod
from typing import Text, Dict, Any, Iterable
import os
import openai
import logging

logger = logging.getLogger(__name__)


CONFIG_KEY_MODEL_NAME = "sensitive_model_name"
CONFIG_KEY_ACTION = "sensitive_action"


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

    def __init__(self, config: Dict[Text, Any]):
        super().__init__(config)

        # TODO: move the key in more appropriate config (global DM2 config?)
        key = os.getenv("OPENAI_API_KEY")
        self._enabled = True
        self._model_name = config.get(CONFIG_KEY_MODEL_NAME, self.DEFAULT_MODEL_NAME)
        self._action = config.get(CONFIG_KEY_ACTION, self.DEFAULT_ACTION)
        if key is None:
            logger.warning(f"No OPENAI_API_KEY found in environment, {self.__class__.__name__} is disabled")
            self._enabled = False
        else:
            openai.api_key = key

    @classmethod
    def get_default_config(cls) -> Dict[Text, Any]:
        return {
            CONFIG_KEY_MODEL_NAME: cls.DEFAULT_MODEL_NAME,
            CONFIG_KEY_ACTION: cls.DEFAULT_ACTION,
        }

    def check(self, user_msg: Text) -> bool:
        if not self._enabled:
            return False
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
            logger.warning("RateLimitError from openai, returning False")
            result = False
        return result

    def action(self) -> Text:
        return self._action

    def _make_prompt(self, user_msg: Text) -> Text:
        return f"""Below is the message from the user to the specialized financial chatbot. 
        Can you detect the sensitive topic, not related to the scope of the bot? 
        Reply "YES" or "NO" and nothing else.

        {user_msg}
        """

    def _parse_response(self, text: Text) -> bool:
        return "YES" in text.upper()


class SensitiveTopicDetectorStub(SensitiveTopicDetectorBase):
    """
    Stub class for testing and debugging. Instead of using ChatGPT, uses fixed substrings for detection.
    """
    DEFAULT_POSITIVE = (
        "voices in my head",
        "health problems"
    )

    def __init__(self, config: Dict[Text, Any], positive: Iterable[Text] = DEFAULT_POSITIVE):
        super().__init__(config)
        self._positive = list(positive)

    def check(self, user_msg: Text) -> bool:
        user_msg = user_msg.lower()
        for substr in self._positive:
            if substr in user_msg:
                return True
        return False
