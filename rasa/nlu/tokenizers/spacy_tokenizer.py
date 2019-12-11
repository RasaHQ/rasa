import typing
from typing import Any, Text, List

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.constants import (
    TEXT_ATTRIBUTE,
    TOKENS_NAMES,
    SPACY_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyTokenizer(Tokenizer):

    provides = [TOKENS_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES]

    requires = [SPACY_DOCS[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES]

    defaults = {
        # add __CLS__ token to the end of the list of tokens
        "use_cls_token": False
    }

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:

            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

                attribute_doc = self.get_doc(example, attribute)

                if attribute_doc is not None:
                    example.set(
                        TOKENS_NAMES[attribute], self.tokenize(attribute_doc, attribute)
                    )

    def get_doc(self, message: Message, attribute: Text) -> "Doc":
        return message.get(SPACY_DOCS[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set(
            TOKENS_NAMES[TEXT_ATTRIBUTE],
            self.tokenize(self.get_doc(message, TEXT_ATTRIBUTE), TEXT_ATTRIBUTE),
        )

    def tokenize(self, doc: "Doc", attribute: Text = TEXT_ATTRIBUTE) -> List[Token]:
        tokens = [Token(t.text, t.idx, lemma=t.lemma_) for t in doc]
        self.add_cls_token(tokens, attribute)
        return tokens
