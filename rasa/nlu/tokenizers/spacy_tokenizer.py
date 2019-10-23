import typing
from typing import Any, Text, List

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

    defaults = {
        # add __CLS__ token to the end of the list of tokens
        "use_cls_token": True
    }

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
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:
            message.set(
                MESSAGE_TOKENS_NAMES[attribute],
                self.tokenize(self.get_doc(message, attribute), attribute),
            )

    def tokenize(
        self, doc: "Doc", attribute: Text = MESSAGE_TEXT_ATTRIBUTE
    ) -> List[Token]:
        tokens = [Token(t.text, t.idx, lemma=t.lemma_) for t in doc]
        self.add_cls_token(tokens, attribute)
        return tokens
