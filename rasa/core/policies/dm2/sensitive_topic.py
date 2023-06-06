from typing import Text
import os
import openai
import logging

logger = logging.getLogger(__name__)


class SensitiveTopicDetector:
    MODEL_NAME = "text-davinci-003"
    TEMPERATURE = 0.0

    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        self._enabled = True
        if key is None:
            logger.warning(f"No OPENAI_API_KEY found in environment, {self.__class__.__name__} is disabled")
            self._enabled = False
        else:
            openai.api_key = key
        pass

    def infer(self, user_msg: Text) -> bool:
        if not self._enabled:
            return False
        try:
            resp = openai.Completion.create(
                model=self.MODEL_NAME,
                prompt=self._make_prompt(user_msg),
                temperature=self.TEMPERATURE,
            )
            resp_text = resp.choices[0].text
            result = self._parse_response(resp_text)
            logger.debug("Response: %s -> %s", resp_text.strip(), result)
        except openai.error.RateLimitError:
            logger.warning("RateLimitError from openai, returning False")
            result = False
        return result

    def _make_prompt(self, user_msg: Text) -> Text:
        return f"""Below is the message from the user to the specialized financial chatbot. 
        Can you detect the sensitive topic, not related to the scope of the bot? 
        Reply "YES" or "NO" and nothing else.

        {user_msg}
        """

    def _parse_response(self, text: Text) -> bool:
        return "YES" in text.upper()
