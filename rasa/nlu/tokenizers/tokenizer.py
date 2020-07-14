import logging
import re

from typing import Text, List, Optional, Dict, Any

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.components import Component
from rasa.nlu.constants import (
    TEXT,
    TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    INTENT,
    ACTION_TEXT,
    MESSAGE_ACTION_NAME,
)

logger = logging.getLogger(__name__)


class Token(object):
    def __init__(
        self,
        text: Text,
        start: int,
        end: Optional[int] = None,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end if end else start + len(text)

        self.data = data if data else {}
        self.lemma = lemma or text

    def set(self, prop: Text, info: Any) -> None:
        self.data[prop] = info

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        return self.data.get(prop, default)

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) == (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __lt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) < (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )


class Tokenizer(Component):
    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

        # flag to check whether to split intents
        self.intent_tokenization_flag = self.component_config.get(
            "intent_tokenization_flag", False
        )
        # split symbol for intents
        self.intent_split_symbol = self.component_config.get("intent_split_symbol", "_")
        # token pattern to further split tokens
        token_pattern = self.component_config.get("token_pattern", None)
        self.token_pattern_regex = None
        if token_pattern:
            self.token_pattern_regex = re.compile(token_pattern)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""

        raise NotImplementedError

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Tokenize all training data."""

        for example in training_data.training_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                if example.get(attribute):
                    if attribute == INTENT:
                        tokens = self._split_intent(example)
                    elif attribute == MESSAGE_ACTION_NAME:
                        # when we have text action, the text will be stored both in `action_name` and in `text`;
                        # we need to check if the text is empty to make sure that it is actually the `action_name`
                        if not example.get(ACTION_TEXT):
                            tokens = self._split_action(example)
                        else:
                            attribute = ACTION_TEXT
                            tokens = self.tokenize(example, attribute)
                    else:
                        tokens = self.tokenize(example, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)

    def process(self, message: Message, attribute: Text = TEXT, **kwargs: Any) -> None:
        """Tokenize the incoming message."""
        if message.get(attribute):
            if attribute == INTENT:
                tokens = self._split_intent(message)
            elif attribute == MESSAGE_ACTION_NAME:
                # when we have text action, the text will be stored both in `action_name` and in `text`;
                # we need to check if the text is empty to make sure that it is actually the `action_name`
                if not message.get(ACTION_TEXT):
                    tokens = self._split_action(message)
                else:
                    attribute = ACTION_TEXT
                    tokens = self.tokenize(message, attribute)
            else:
                tokens = self.tokenize(message, attribute)

            message.set(TOKENS_NAMES[attribute], tokens)

    def _split_intent(self, message: Message) -> List[Token]:
        text = message.get(INTENT)

        words = (
            text.split(self.intent_split_symbol)
            if self.intent_tokenization_flag
            else [text]
        )

        return self._convert_words_to_tokens(words, text)

    def _apply_token_pattern(self, tokens: List[Token]) -> List[Token]:
        """Apply the token pattern to the given tokens.

        Args:
            tokens: list of tokens to split

        Returns:
            List of tokens.
        """
        if not self.token_pattern_regex:
            return tokens

        final_tokens = []
        for token in tokens:
            new_tokens = self.token_pattern_regex.findall(token.text)
            new_tokens = [t for t in new_tokens if t]

            if not new_tokens:
                final_tokens.append(token)

            running_offset = 0
            for new_token in new_tokens:
                word_offset = token.text.index(new_token, running_offset)
                word_len = len(new_token)
                running_offset = word_offset + word_len
                final_tokens.append(
                    Token(
                        new_token,
                        token.start + word_offset,
                        data=token.data,
                        lemma=token.lemma,
                    )
                )

        return final_tokens

    def _split_action(self, message: Message) -> List[Token]:
        if message.get(MESSAGE_ACTION_NAME):
            text = message.get(MESSAGE_ACTION_NAME)
        else:
            # during processing we store action name in text field
            text = message.get(TEXT)
        # TODO: Do we want a separate `action_split_symbol` or same as intent?
        words = text.split(self.intent_split_symbol)

        return self._convert_words_to_tokens(words, text)

    @staticmethod
    def _convert_words_to_tokens(words: List[Text], text: Text) -> List[Token]:
        running_offset = 0
        tokens = []

        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))

        return tokens
