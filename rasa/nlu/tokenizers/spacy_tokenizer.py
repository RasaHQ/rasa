import typing
from typing import Any, Dict, Text, List, Optional

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_SPACY_FEATURES_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
    CLS_TOKEN,
)

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc  # pytype: disable=import-error


class SpacyTokenizer(Tokenizer, Component):

    provides = [
        MESSAGE_TOKENS_NAMES[attribute] for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    requires = [
        MESSAGE_SPACY_FEATURES_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {
        # Add a __cls__ token to the end of the list of tokens
        "add_cls_token": False
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        """Construct a new tokenizer using the SpacyTokenizer framework."""
        super(SpacyTokenizer, self).__init__(component_config)
        self.add_cls_token = self.component_config["add_cls_token"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.training_examples:

            for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:

                attribute_doc = self.get_doc(example, attribute)

                if attribute_doc is not None:
                    example.set(
                        MESSAGE_TOKENS_NAMES[attribute], self.tokenize(attribute_doc)
                    )

    def get_doc(self, message: Message, attribute: Text) -> "Doc":
        return message.get(MESSAGE_SPACY_FEATURES_NAMES[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set(
            MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self.tokenize(self.get_doc(message, MESSAGE_TEXT_ATTRIBUTE)),
        )

    def tokenize(self, doc: "Doc") -> List[Token]:
        tokens = [Token(t.text, t.idx) for t in doc]
        if self.add_cls_token:
            idx = doc[-1].idx + len(doc[-1].text) + 1
            tokens = tokens + [Token(CLS_TOKEN, idx)]
        return tokens
