from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Any, List

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyTokenizer(Tokenizer, Component):
    name = "tokenizer_spacy"

    provides = ["tokens"]

    requires = ["spacy_doc"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.get("spacy_doc")))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.get("spacy_doc")))

    def tokenize(self, doc):
        # type: (Doc) -> List[Token]

        return [Token(t.text, t.idx) for t in doc]
