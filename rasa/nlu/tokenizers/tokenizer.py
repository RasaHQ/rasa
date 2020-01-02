import logging

from typing import Text, List, Optional, Dict, Any

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.components import Component
from rasa.nlu.constants import (
    RESPONSE_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    CLS_TOKEN,
    TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
)

logger = logging.getLogger(__name__)


class Token(object):
    def __init__(
        self,
        text: Text,
        start: int,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
        end: Optional[int] = None,
    ) -> None:
        self.start = start
        self.text = text
        self.end = start + len(text)
        self.data = data if data else {}
        self.lemma = lemma or text
        self.end = end if end else start + len(text)

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
    def train_attributes(self) -> List[Text]:
        """Returns the attributes of a message that indicate what kind of texts should
        be tokenized."""

        return MESSAGE_ATTRIBUTES

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
            for attribute in self.train_attributes():
                if example.get(attribute) is not None:
                    tokens = self.tokenize(example, attribute)
                    tokens = self.add_cls_token(tokens, attribute)
                    example.set(TOKENS_NAMES[attribute], tokens)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Tokenize the incoming message."""

        tokens = self.tokenize(message, TEXT_ATTRIBUTE)
        tokens = self.add_cls_token(tokens, TEXT_ATTRIBUTE)
        message.set(TOKENS_NAMES[TEXT_ATTRIBUTE], tokens)

    def add_cls_token(self, tokens: List[Token], attribute: Text) -> List[Token]:
        if attribute in [RESPONSE_ATTRIBUTE, TEXT_ATTRIBUTE] and tokens:
            # +1 to have a space between the last token and the __cls__ token
            idx = tokens[-1].end + 1
            tokens.append(Token(CLS_TOKEN, idx))

        return tokens
