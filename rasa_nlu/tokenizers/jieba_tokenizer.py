from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import logging

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message, TrainingData
from typing import Any, List, Text

logger = logging.getLogger(__name__)


class JiebaTokenizer(Tokenizer, Component):
    name = "tokenizer_jieba"

    provides = ["tokens"]

    language_list = ["zh"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["jieba"]

    @staticmethod
    def load_custom_dictionary(path):
        # type: (Text) -> None
        """Load all the custom dictionaries stored in the path.

        More information about the dictionaries file format can
        be found in the documentation of jieba.
        https://github.com/fxsjy/jieba#load-dictionary
        """
        import jieba

        jieba_userdicts = glob.glob("{}/*".format(path))
        for jieba_userdict in jieba_userdicts:
            logger.info("Loading Jieba User Dictionary at "
                        "{}".format(jieba_userdict))
            jieba.load_userdict(jieba_userdict)

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        import jieba
        tokenized = jieba.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]
        return tokens
