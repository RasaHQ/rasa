from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import logging
import os
import shutil

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message, TrainingData
from typing import Any, List, Text

logger = logging.getLogger(__name__)

JIEBA_CUSTOM_DICTIONARY_PATH = "tokenizer_jieba"


class JiebaTokenizer(Tokenizer, Component):
    name = "tokenizer_jieba"

    provides = ["tokens"]

    language_list = ["zh"]

    defaults = {
        "dictionary_path": None  # default don't load custom dictionary
    }

    def __init__(self, component_config=None):
        # type: (Dict[Text, Any]) -> None
        """Construct a new intent classifier using the MITIE framework."""

        super(JiebaTokenizer, self).__init__(component_config)

        # path to dictionary file or None
        self.dictionary_path = self.component_config.get('dictionary_path')

        # load dictionary
        if self.dictionary_path is not None:
            self.load_custom_dictionary(self.dictionary_path)

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

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> JiebaTokenizer

        meta = model_metadata.for_component(cls.name)
        relative_dictionary_path = meta.get("dictionary_path")

        # get real path of dictionary path, if any
        if relative_dictionary_path is not None:
            dictionary_path = os.path.join(model_dir, relative_dictionary_path)

            meta["dictionary_path"] = dictionary_path

        return cls(meta)

    @staticmethod
    def copy_files_dir_to_dir(input_dir, output_dir):
        # make sure target path exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob("{}/*".format(input_dir))
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory."""

        model_dictionary_path = None

        # copy custom dictionaries to model dir, if any
        if self.dictionary_path is not None:
            target_dictionary_path = os.path.join(model_dir,
                                                  JIEBA_CUSTOM_DICTIONARY_PATH)
            self.copy_files_dir_to_dir(self.dictionary_path,
                                       target_dictionary_path)

            # set dictionary_path of model metadata to relative path
            model_dictionary_path = JIEBA_CUSTOM_DICTIONARY_PATH

        return {"dictionary_path": model_dictionary_path}
