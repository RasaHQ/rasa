import logging
import re

from typing import Text, List, Optional, Dict, Any

from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import (
    TrainingDataFull,
    TrainingDataChunk,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.components import Component
from rasa.nlu.constants import MESSAGE_ATTRIBUTES
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_RESPONSE_KEY,
    RESPONSE_IDENTIFIER_DELIMITER,
    ACTION_NAME,
    TOKENS_NAMES,
)
from rasa.shared.exceptions import RasaTrainChunkException

logger = logging.getLogger(__name__)


class Token:
    """Represents a token in a text."""

    def __init__(
        self,
        text: Text,
        start: int,
        end: Optional[int] = None,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
    ) -> None:
        """Initializes the token.

        Args:
            text: The text of the token.
            start: The start value of the token in the complete sentence.
            end: The end value of the token in the complete sentence.
            data: Optional data.
            lemma: The lemma of the token.
        """
        self.text = text
        self.start = start
        self.end = end if end else start + len(text)

        self.data = data if data else {}
        self.lemma = lemma or text

    def set(self, prop: Text, info: Any) -> None:
        """Sets a certain property inside data.

        Args:
            prop: The property to set.
            info: The value to set 'prop' to.
        """
        self.data[prop] = info

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        """Gets a certain property from data.

        Args:
            prop: The property to return.
            default: The default value in case 'prop' is not present.

        Returns:
            The value of the property.
        """
        return self.data.get(prop, default)

    @classmethod
    def from_dict(cls, token_dict: Dict[Text, Any]) -> "Token":
        """Creates a token from the given dict.

        Args:
            token_dict: the dictionary that contains all properties of a token.

        Returns:
            A token.
        """
        return Token(
            token_dict["text"],
            token_dict["start"],
            token_dict["end"] if "end" in token_dict else None,
            token_dict["data"] if "data" in token_dict else None,
            token_dict["lemma"] if "lemma" in token_dict else None,
        )

    def __eq__(self, other: Any) -> bool:
        """Compares this token to another token."""
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

    def __str__(self) -> Text:
        """Return the string representation of this Token."""
        return f"{self.text} ({self.start}-{self.end})"


class Tokenizer(Component):
    """Abstract tokenizer component."""

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

    def train_chunk(
        self,
        training_data_chunk: TrainingDataChunk,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component on the given chunk.

        See parent class for more information.
        """
        raise RasaTrainChunkException(
            "This method should neither be called nor implemented in our code."
        )

    def train(
        self,
        training_data: TrainingDataFull,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Tokenize all training data.

        Args:
            training_data: the
                :class:`rasa.shared.nlu.training_data.training_data.TrainingData`.
            config: The model configuration parameters.
        """
        for example in training_data.training_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                if (
                    example.get(attribute) is not None
                    and not example.get(attribute) == ""
                ):
                    if attribute in [INTENT, ACTION_NAME, INTENT_RESPONSE_KEY]:
                        tokens = self._split_name(example, attribute)
                    else:
                        tokens = self.tokenize(example, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Tokenize the incoming message."""
        for attribute in MESSAGE_ATTRIBUTES:
            if isinstance(message.get(attribute), str):
                if attribute in [INTENT, ACTION_NAME, RESPONSE_IDENTIFIER_DELIMITER]:
                    tokens = self._split_name(message, attribute)
                else:
                    tokens = self.tokenize(message, attribute)

                message.set(TOKENS_NAMES[attribute], tokens)

    def _tokenize_on_split_symbol(self, text: Text) -> List[Text]:

        words = (
            text.split(self.intent_split_symbol)
            if self.intent_tokenization_flag
            else [text]
        )

        return words

    def _split_name(self, message: Message, attribute: Text = INTENT) -> List[Token]:
        text = message.get(attribute)

        # for INTENT_RESPONSE_KEY attribute,
        # first split by RESPONSE_IDENTIFIER_DELIMITER
        if attribute == INTENT_RESPONSE_KEY:
            intent, response_key = text.split(RESPONSE_IDENTIFIER_DELIMITER)
            words = self._tokenize_on_split_symbol(
                intent
            ) + self._tokenize_on_split_symbol(response_key)

        else:
            words = self._tokenize_on_split_symbol(text)

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
