import logging

from typing import Any, Dict, List, NoReturn, Optional, Text, Tuple, Type
from tqdm import tqdm
import os

import rasa.shared.utils.io
import rasa.core.utils
from rasa.utils import common
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.model import Metadata
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
    NUMBER_OF_SUB_TOKENS,
)
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_TEXT,
)
from rasa.exceptions import RasaException
import rasa.nlu.utils
import numpy as np
import tensorflow as tf

import rasa.utils.train_utils as train_utils

logger = logging.getLogger(__name__)

# URL to the old remote location of the model which
# users might use. The model is no longer hosted here.
ORIGINAL_TF_HUB_MODULE_URL = (
    "https://github.com/PolyAI-LDN/polyai-models/releases/download/v1.0/model.tar.gz"
)

# Warning: This URL is only intended for running pytests on ConveRT
# related components. This URL should not be allowed to be used by the user.
RESTRICTED_ACCESS_URL = "https://storage.googleapis.com/continuous-integration-model-storage/convert_tf2.tar.gz"


class ConveRTFeaturizer(DenseFeaturizer):
    """Featurizer using ConveRT model.

    Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
    model from TFHub and computes sentence and sequence level feature representations
    for dense featurizable attributes of each message object.
    """

    defaults = {
        # Remote URL/Local path to model files
        "model_url": None
    }

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        """Packages needed to be installed."""
        return ["tensorflow_text", "tensorflow_hub"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        """Initializes ConveRTFeaturizer with the model and different
        encoding signatures.

        Args:
            component_config: Configuration for the component.
        """
        super(ConveRTFeaturizer, self).__init__(component_config)
        self.model_url = self._get_validated_model_url()

        self.module = train_utils.load_tf_hub_model(self.model_url)

        self.tokenize_signature = self._get_signature("tokenize", self.module)
        self.sequence_encoding_signature = self._get_signature(
            "encode_sequence", self.module
        )
        self.sentence_encoding_signature = self._get_signature("default", self.module)

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
                of "{ConveRTFeaturizer.__name__}". It is mandatory to pass a value for this parameter.
                You can either use a community hosted URL of the model
                or if you have a local copy of the model, pass the
                path to the directory containing the model files."""
            )

        if model_url == ORIGINAL_TF_HUB_MODULE_URL:
            # Can't use the originally hosted URL
            raise RasaException(
                f"""Parameter "model_url" of "{ConveRTFeaturizer.__name__}" was
                set to "{model_url}" which does not contain the model any longer.
                You can either use a community hosted URL or if you have a
                local copy of the model, pass the path to the directory
                containing the model files."""
            )

        if model_url == RESTRICTED_ACCESS_URL:
            # Can't use the URL that is reserved for tests only
            raise RasaException(
                f"""Parameter "model_url" of "{ConveRTFeaturizer.__name__}" was
                set to "{model_url}" which is strictly reserved for pytests of Rasa Open Source only.
                Due to licensing issues you are not allowed to use the model from this URL.
                You can either use a community hosted URL or if you have a
                local copy of the model, pass the path to the directory
                containing the model files."""
            )

        if os.path.isfile(model_url):
            # Definitely invalid since the specified path should be a directory
            raise RasaException(
                f"""Parameter "model_url" of "{ConveRTFeaturizer.__name__}" was
                set to the path of a file which is invalid. You
                can either use a community hosted URL or if you have a
                local copy of the model, pass the path to the directory
                containing the model files."""
            )

        if rasa.nlu.utils.is_url(model_url):
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

    @staticmethod
    def _get_signature(signature: Text, module: Any) -> NoReturn:
        """Retrieve a signature from a (hopefully loaded) TF model."""
        if not module:
            raise Exception(
                "ConveRTFeaturizer needs a proper loaded tensorflow module when used. "
                "Make sure to pass a module when training and using the component."
            )

        return module.signatures[signature]

    def _compute_features(
        self, batch_examples: List[Message], attribute: Text = TEXT
    ) -> Tuple[np.ndarray, np.ndarray]:
        sentence_encodings = self._compute_sentence_encodings(batch_examples, attribute)

        (
            sequence_encodings,
            number_of_tokens_in_sentence,
        ) = self._compute_sequence_encodings(batch_examples, attribute)

        return self._get_features(
            sentence_encodings, sequence_encodings, number_of_tokens_in_sentence
        )

    def _compute_sentence_encodings(
        self, batch_examples: List[Message], attribute: Text = TEXT
    ) -> np.ndarray:
        # Get text for attribute of each example
        batch_attribute_text = [ex.get(attribute) for ex in batch_examples]
        sentence_encodings = self._sentence_encoding_of_text(batch_attribute_text)

        # convert them to a sequence of 1
        return np.reshape(sentence_encodings, (len(batch_examples), 1, -1))

    def _compute_sequence_encodings(
        self, batch_examples: List[Message], attribute: Text = TEXT
    ) -> Tuple[np.ndarray, List[int]]:
        list_of_tokens = [
            self.tokenize(example, attribute) for example in batch_examples
        ]

        number_of_tokens_in_sentence = [
            len(sent_tokens) for sent_tokens in list_of_tokens
        ]

        # join the tokens to get a clean text to ensure the sequence length of
        # the returned embeddings from ConveRT matches the length of the tokens
        # (including sub-tokens)
        tokenized_texts = self._tokens_to_text(list_of_tokens)
        token_features = self._sequence_encoding_of_text(tokenized_texts)

        # ConveRT might split up tokens into sub-tokens
        # take the mean of the sub-token vectors and use that as the token vector
        token_features = train_utils.align_token_features(
            list_of_tokens, token_features
        )

        return token_features, number_of_tokens_in_sentence

    @staticmethod
    def _get_features(
        sentence_encodings: np.ndarray,
        sequence_encodings: np.ndarray,
        number_of_tokens_in_sentence: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the sequence and sentence features."""
        sentence_embeddings = []
        sequence_embeddings = []

        for index in range(len(number_of_tokens_in_sentence)):
            sequence_length = number_of_tokens_in_sentence[index]
            sequence_encoding = sequence_encodings[index][:sequence_length]
            sentence_encoding = sentence_encodings[index]

            sequence_embeddings.append(sequence_encoding)
            sentence_embeddings.append(sentence_encoding)

        return np.array(sequence_embeddings), np.array(sentence_embeddings)

    @staticmethod
    def _tokens_to_text(list_of_tokens: List[List[Token]]) -> List[Text]:
        """Convert list of tokens to text.

        Add a whitespace between two tokens if the end value of the first tokens
        is not the same as the end value of the second token.
        """
        texts = []
        for tokens in list_of_tokens:
            text = ""
            offset = 0
            for token in tokens:
                if offset != token.start:
                    text += " "
                text += token.text

                offset = token.end
            texts.append(text)

        return texts

    def _sentence_encoding_of_text(self, batch: List[Text]) -> np.ndarray:

        return self.sentence_encoding_signature(tf.convert_to_tensor(batch))[
            "default"
        ].numpy()

    def _sequence_encoding_of_text(self, batch: List[Text]) -> np.ndarray:

        return self.sequence_encoding_signature(tf.convert_to_tensor(batch))[
            "sequence_encoding"
        ].numpy()

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Featurize all message attributes in the training data with the ConveRT model.

        Args:
            training_data: Training data to be featurized
            config: Pipeline configuration
            **kwargs: Any other arguments.
        """
        if config is not None and config.language != "en":
            rasa.shared.utils.io.raise_warning(
                f"Since ``ConveRT`` model is trained only on an english "
                f"corpus of conversations, this featurizer should only be "
                f"used if your training data is in english language. "
                f"However, you are training in '{config.language}'. ",
                docs=DOCS_URL_COMPONENTS + "#convertfeaturizer",
            )

        batch_size = 64

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            )

            progress_bar = tqdm(
                range(0, len(non_empty_examples), batch_size),
                desc=attribute.capitalize() + " batches",
            )
            for batch_start_index in progress_bar:
                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )

                # Collect batch examples
                batch_examples = non_empty_examples[batch_start_index:batch_end_index]

                (
                    batch_sequence_features,
                    batch_sentence_features,
                ) = self._compute_features(batch_examples, attribute)

                self._set_features(
                    batch_examples,
                    batch_sequence_features,
                    batch_sentence_features,
                    attribute,
                )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Featurize an incoming message with the ConveRT model.

        Args:
            message: Message to be featurized
            **kwargs: Any other arguments.
        """
        for attribute in {TEXT, ACTION_TEXT}:
            if message.get(attribute):
                sequence_features, sentence_features = self._compute_features(
                    [message], attribute=attribute
                )

                self._set_features(
                    [message], sequence_features, sentence_features, attribute
                )

    def _set_features(
        self,
        examples: List[Message],
        sequence_features: np.ndarray,
        sentence_features: np.ndarray,
        attribute: Text,
    ) -> None:
        for index, example in enumerate(examples):
            _sequence_features = Features(
                sequence_features[index],
                FEATURE_TYPE_SEQUENCE,
                attribute,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            )
            example.add_features(_sequence_features)

            _sentence_features = Features(
                sentence_features[index],
                FEATURE_TYPE_SENTENCE,
                attribute,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            )
            example.add_features(_sentence_features)

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
        return f"{cls.name}-{rasa.shared.utils.io.deep_container_fingerprint(_config)}"

    def provide_context(self) -> Dict[Text, Any]:
        """Store the model in pipeline context for future use."""
        return {"tf_hub_module": self.module}

    def _tokenize(self, sentence: Text) -> Any:

        return self.tokenize_signature(tf.convert_to_tensor([sentence]))[
            "default"
        ].numpy()

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        """Tokenize the text using the ConveRT model.

        ConveRT adds a special char in front of (some) words and splits words into
        sub-words. To ensure the entity start and end values matches the token values,
        reuse the tokens that are already assigned to the message. If individual tokens
        are split up into multiple tokens, add this information to the
        respected tokens.
        """
        tokens_in = message.get(TOKENS_NAMES[attribute])

        tokens_out = []

        for token in tokens_in:
            # use ConveRT model to tokenize the text
            split_token_strings = self._tokenize(token.text)[0]

            # clean tokens (remove special chars and empty tokens)
            split_token_strings = self._clean_tokens(split_token_strings)

            token.set(NUMBER_OF_SUB_TOKENS, len(split_token_strings))

            tokens_out.append(token)

        message.set(TOKENS_NAMES[attribute], tokens_out)
        return tokens_out

    @staticmethod
    def _clean_tokens(tokens: List[bytes]) -> List[Text]:
        """Encode tokens and remove special char added by ConveRT."""
        tokens = [string.decode("utf-8").replace("﹏", "") for string in tokens]
        return [string for string in tokens if string]
