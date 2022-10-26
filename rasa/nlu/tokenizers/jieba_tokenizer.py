from __future__ import annotations
import glob
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER, is_trainable=True
)
class JiebaTokenizer(Tokenizer):
    """This tokenizer is a wrapper for Jieba (https://github.com/fxsjy/jieba)."""

    @staticmethod
    def supported_languages() -> Optional[List[Text]]:
        """Supported languages (see parent class for full docstring)."""
        return ["zh"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default config (see parent class for full docstring)."""
        return {
            # default don't load custom dictionary
            "dictionary_path": None,
            # Flag to check whether to split intents
            "intent_tokenization_flag": False,
            # Symbol on which intent should be split
            "intent_split_symbol": "_",
            # Regular expression to detect tokens
            "token_pattern": None,
        }

    def __init__(
        self, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource
    ) -> None:
        """Initialize the tokenizer."""
        super().__init__(config)
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> JiebaTokenizer:
        """Creates a new component (see parent class for full docstring)."""
        # Path to the dictionaries on the local filesystem.
        dictionary_path = config["dictionary_path"]

        if dictionary_path is not None:
            cls._load_custom_dictionary(dictionary_path)
        return cls(config, model_storage, resource)

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["jieba"]

    @staticmethod
    def _load_custom_dictionary(path: Text) -> None:
        """Load all the custom dictionaries stored in the path.

        More information about the dictionaries file format can
        be found in the documentation of jieba.
        https://github.com/fxsjy/jieba#load-dictionary
        """
        import jieba

        jieba_userdicts = glob.glob(f"{path}/*")
        for jieba_userdict in jieba_userdicts:
            logger.info(f"Loading Jieba User Dictionary at {jieba_userdict}")
            jieba.load_userdict(jieba_userdict)

    def train(self, training_data: TrainingData) -> Resource:
        """Copies the dictionary to the model storage."""
        self.persist()
        return self._resource

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenizes the text of the provided attribute of the incoming message."""
        import jieba

        text = message.get(attribute)

        tokenized = jieba.tokenize(text)
        tokens = [Token(word, start) for (word, start, end) in tokenized]

        return self._apply_token_pattern(tokens)

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> JiebaTokenizer:
        """Loads a custom dictionary from model storage."""
        dictionary_path = config["dictionary_path"]

        # If a custom dictionary path is in the config we know that it should have
        # been saved to the model storage.
        if dictionary_path is not None:
            try:
                with model_storage.read_from(resource) as resource_directory:
                    cls._load_custom_dictionary(str(resource_directory))
            except ValueError:
                logger.debug(
                    f"Failed to load {cls.__name__} from model storage. "
                    f"Resource '{resource.name}' doesn't exist."
                )
        return cls(config, model_storage, resource)

    @staticmethod
    def _copy_files_dir_to_dir(input_dir: Text, output_dir: Text) -> None:
        # make sure target path exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob(f"{input_dir}/*")
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)

    def persist(self) -> None:
        """Persist the custom dictionaries."""
        dictionary_path = self._config["dictionary_path"]
        if dictionary_path is not None:
            with self._model_storage.write_to(self._resource) as resource_directory:
                self._copy_files_dir_to_dir(dictionary_path, str(resource_directory))
