import typing
from typing import Any

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.constants import (
    RESPONSE_ATTRIBUTE,
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    TOKEN_NAMES,
    ATTRIBUTES,
    SPACY_FEATURE_NAMES,
    FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyTokenizer(Tokenizer, Component):

    provides = [TOKEN_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES]

    requires = [
        SPACY_FEATURE_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES
    ]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:

            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

                attribute_doc = self.get_doc(example, attribute)

                if attribute_doc is not None:
                    example.set(TOKEN_NAMES[attribute], self.tokenize(attribute_doc))

    def get_doc(self, message, attribute):

        return message.get(SPACY_FEATURE_NAMES[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set(
            TOKEN_NAMES[TEXT_ATTRIBUTE],
            self.tokenize(self.get_doc(message, TEXT_ATTRIBUTE)),
        )

    def tokenize(self, doc: "Doc") -> typing.List[Token]:

        return [Token(t.text, t.idx) for t in doc]
