from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class HanlpTokenizer(Tokenizer, Component):
    name = "tokenizer_hanlp"

    provides = ["tokens"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["pyhanlp"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        from pyhanlp import HanLP
        terms = HanLP.segment(text)
        running_offset = 0
        tokens = []
        for term in terms:
            word_offset = text.index(term.word, running_offset)
            word_len = len(term.word)
            running_offset = word_offset + word_len
            tokens.append(Token(term.word, word_offset))
        return tokens
