import typing
from typing import Any, Dict, Text, List, Optional

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_SPACY_FEATURES_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyTokenizer(Tokenizer):

    provides = [
        MESSAGE_TOKENS_NAMES[attribute] for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    requires = [
        MESSAGE_SPACY_FEATURES_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:

            for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:

                attribute_doc = self.get_doc(example, attribute)

                if attribute_doc is not None:
                    example.set(
                        MESSAGE_TOKENS_NAMES[attribute],
                        self.tokenize(attribute_doc, attribute),
                    )

    def get_doc(self, message: Message, attribute: Text) -> "Doc":
        return message.get(MESSAGE_SPACY_FEATURES_NAMES[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set(
            MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self.tokenize(
                self.get_doc(message, MESSAGE_TEXT_ATTRIBUTE), MESSAGE_TEXT_ATTRIBUTE
            ),
        )

    def tokenize(self, doc: "Doc", attribute: Text) -> List[Token]:
        tokens = [Token(t.text, t.idx) for t in doc]
        self.add_cls_token(tokens, attribute)
        return tokens
