from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import str
import re

from typing import Any
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.tokenizers import Token
from rasa_nlu.tokenizers import Tokenizer
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class JiebaTokenizer(Tokenizer, Component):
    name = "tokenizer_jieba"

    provides = ["tokens"]

    def __init__(self):
        pass

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["jieba"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))
    
    def tokenize_with_offsets(self, text):
        # type: (Text) -> Tuple[List[Text], List[int]]
        import jieba

        _text = text.encode('utf-8')
        offsets = []
        offset = 0
        tokens = [w for w in jieba.lcut(_text)]
        for tok in tokens:
            m = re.search(re.escape(tok), text[offset:], re.UNICODE)
            if m is None:
                message = "Invalid Jieba offset. Token '{}' in message '{}'.".format(str(tok),str(text.encode('utf-8')))
                raise ValueError(message)
            offsets.append(offset + m.start())
            offset += m.end()
        return tokens, offsets

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        _text = text.encode('utf-8')
        tokenized = self.tokenize_with_offsets(_text)
        tokens = [Token(token.decode('utf-8'), self._byte_to_char_offset(_text, offset)) for token, offset in tokenized]
        return tokens

    def _byte_to_char_offset(self, text, byte_offset):
        return len(text[:byte_offset].decode('utf-8'))
