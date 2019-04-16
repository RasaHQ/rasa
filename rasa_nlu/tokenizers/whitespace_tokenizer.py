import re
from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Token, Tokenizer
from rasa_nlu.training_data import Message, TrainingData


class WhitespaceTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    def train(self, training_data: TrainingData, config: RasaNLUModelConfig,
              **kwargs: Any) -> None:

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text: Text) -> List[Token]:

        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        words = re.sub(r'[.,!?]+(\s|$)', ' ', text).split()

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens
