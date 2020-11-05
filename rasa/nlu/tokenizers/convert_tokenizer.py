from typing import Any, Dict, List, Optional, Text

from rasa.core.utils import get_dict_hash
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.message import Message
from rasa.utils import common
from rasa.nlu import utils as nlu_utils
import rasa.utils.train_utils as train_utils
from rasa.exceptions import RasaException
import tensorflow as tf
import os


# URL to the old remote location of the model which
# users might use. The model is no longer hosted here.
ORIGINAL_TF_HUB_MODULE_URL = (
    "https://github.com/PolyAI-LDN/polyai-models/releases/download/v1.0/model.tar.gz"
)

# Warning: This URL is only intended for running pytests on ConveRT
# related components. This URL should not be allowed to be used by the user.
RESTRICTED_ACCESS_URL = "https://storage.googleapis.com/continuous-integration-model-storage/convert_tf2.tar.gz"


class ConveRTTokenizer(WhitespaceTokenizer):
    """Tokenizer using ConveRT model.

    Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
    model from TFHub and computes sub-word tokens for dense
    featurizable attributes of each message object.
    """

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
        # Remote URL/Local path to model files
        "model_url": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework.

        Args:
            component_config: User configuration for the component
        """
        super().__init__(component_config)

        self.model_url = self._get_validated_model_url()

        self.module = train_utils.load_tf_hub_model(self.model_url)

        self.tokenize_signature = self.module.signatures["tokenize"]

    @staticmethod
    def _validate_model_files_exist(model_directory: Text) -> None:
        """Check if essential model files exist inside the model_directory.

        Args:
            model_directory: Directory to investigate
        """
        files_to_check = [
            os.path.join(model_directory, "saved_model.pb"),
            os.path.join(model_directory, "variables/variables.index"),
            os.path.join(model_directory, "variables/variables.data-00001-of-00002"),
            os.path.join(model_directory, "variables/variables.data-00000-of-00002"),
        ]

        for file_path in files_to_check:
            if not os.path.exists(file_path):
                raise RasaException(
                    f"""File {file_path} does not exist.
                    Re-check the files inside the directory {model_directory}.
                    It should contain the following model
                    files - [{", ".join(files_to_check)}]"""
                )

    def _get_validated_model_url(self) -> Text:
        """Validates the specified `model_url` parameter.

        The `model_url` parameter cannot be left empty. It can either
        be set to a remote URL where the model is hosted or it can be
        a path to a local directory.

        Returns:
            Validated path to model
        """
        model_url = self.component_config.get("model_url", None)

        if not model_url:
            raise RasaException(
                f"""Parameter "model_url" was not specified in the configuration
                of "{ConveRTTokenizer.__name__}".
                You can either use a community hosted URL of the model
                or if you have a local copy of the model, pass the
                path to the directory containing the model files."""
            )

        if model_url == ORIGINAL_TF_HUB_MODULE_URL:
            # Can't use the originally hosted URL
            raise RasaException(
                f"""Parameter "model_url" of "{ConveRTTokenizer.__name__}" was
                set to "{model_url}" which does not contain the model any longer.
                You can either use a community hosted URL or if you have a
                local copy of the model, pass the path to the directory
                containing the model files."""
            )

        if model_url == RESTRICTED_ACCESS_URL:
            # Can't use the URL that is reserved for tests only
            raise RasaException(
                f"""Parameter "model_url" of "{ConveRTTokenizer.__name__}" was
                set to "{model_url}" which is strictly reserved for pytests of Rasa Open Source only.
                Due to licensing issues you are not allowed to use the model from this URL.
                You can either use a community hosted URL or if you have a
                local copy of the model, pass the path to the directory
                containing the model files."""
            )

        if os.path.isfile(model_url):
            # Definitely invalid since the specified path should be a directory
            raise RasaException(
                f"""Parameter "model_url" of "{ConveRTTokenizer.__name__}" was
                set to the path of a file which is invalid. You
                can either use a community hosted URL or if you have a
                local copy of the model, pass the path to the directory
                containing the model files."""
            )

        if nlu_utils.is_url(model_url):
            return model_url

        if os.path.isdir(model_url):
            # Looks like a local directory. Inspect the directory
            # to see if model files exist.
            self._validate_model_files_exist(model_url)
            # Convert the path to an absolute one since
            # TFHUB doesn't like relative paths
            return os.path.abspath(model_url)

        raise RasaException(
            f"""{model_url} is neither a valid remote URL nor a local directory.
            You can either use a community hosted URL or if you have a
            local copy of the model, pass the path to
            the directory containing the model files."""
        )

    @classmethod
    def cache_key(
        cls, component_meta: Dict[Text, Any], model_metadata: Metadata
    ) -> Optional[Text]:
        """Cache the component for future use.

        Args:
            component_meta: configuration for the component.
            model_metadata: configuration for the whole pipeline.

        Returns: key of the cache for future retrievals.
        """
        _config = common.update_existing_keys(cls.defaults, component_meta)
        return f"{cls.name}-{get_dict_hash(_config)}"

    def provide_context(self) -> Dict[Text, Any]:
        return {"tf_hub_module": self.module}

    def _tokenize(self, sentence: Text) -> Any:

        return self.tokenize_signature(tf.convert_to_tensor([sentence]))[
            "default"
        ].numpy()

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the ConveRT model.
        ConveRT adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        tokenize the text first using the whitespace tokenizer. If individual tokens
        are split up into multiple tokens, add this information to the
        respected tokens.
        """

        # perform whitespace tokenization
        tokens_in = super().tokenize(message, attribute)

        tokens_out = []

        for token in tokens_in:
            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token.text)[0]

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        return tokens_out

    @staticmethod
    def _clean_tokens(tokens: List[bytes]) -> List[Text]:
        """Encode tokens and remove special char added by ConveRT."""

        tokens = [string.decode("utf-8").replace("﹏", "") for string in tokens]
        return [string for string in tokens if string]
