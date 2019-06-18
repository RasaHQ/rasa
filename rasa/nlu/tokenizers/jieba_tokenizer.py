import glob
import logging
import os
import shutil
import typing
from typing import Any, Dict, List, Optional, Text

from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers import Token, Tokenizer
from rasa.nlu.training_data import Message, TrainingData

logger = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class JiebaTokenizer(Tokenizer, Component):

    provides = ["tokens"]

    language_list = ["zh"]

    defaults = {"dictionary_path": None}  # default don't load custom dictionary

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new intent classifier using the MITIE framework."""

        super(JiebaTokenizer, self).__init__(component_config)

        # path to dictionary file or None
        self.dictionary_path = self.component_config.get("dictionary_path")

        # load dictionary
        if self.dictionary_path is not None:
            self.load_custom_dictionary(self.dictionary_path)

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["jieba"]

    @staticmethod
    def load_custom_dictionary(path: Text) -> None:
        """Load all the custom dictionaries stored in the path.

        More information about the dictionaries file format can
        be found in the documentation of jieba.
        https://github.com/fxsjy/jieba#load-dictionary
        """
        import jieba

        jieba_userdicts = glob.glob("{}/*".format(path))
        for jieba_userdict in jieba_userdicts:
            logger.info("Loading Jieba User Dictionary at {}".format(jieba_userdict))
            jieba.load_userdict(jieba_userdict)

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message: Message, **kwargs: Any) -> None:
        message.set("tokens", self.tokenize(message.text))

    @staticmethod
    def tokenize(text: Text) -> List[Token]:
        import jieba

        tokenized = jieba.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]
        return tokens

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional[Component] = None,
        **kwargs: Any
    ) -> "JiebaTokenizer":

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

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        # copy custom dictionaries to model dir, if any
        if self.dictionary_path is not None:
            target_dictionary_path = os.path.join(model_dir, file_name)
            self.copy_files_dir_to_dir(self.dictionary_path, target_dictionary_path)

            return {"dictionary_path": file_name}
        else:
            return {"dictionary_path": None}
