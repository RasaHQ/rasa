import logging

from tensorflow_core.python.training.tracking.tracking import AutoTrackable
from typing import Any, Dict, List, NoReturn, Optional, Text, Tuple, Type
from tqdm import tqdm

from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES, DENSE_FEATURIZABLE_ATTRIBUTES
import numpy as np
import tensorflow as tf

import rasa.utils.train_utils as train_utils
import rasa.utils.common as common_utils

logger = logging.getLogger(__name__)


class ConveRTFeaturizer(DenseFeaturizer):
    """Featurizer using ConveRT model.

    Loads the ConveRT(https://github.com/PolyAI-LDN/polyai-models#convert)
    model from TFHub and computes sentence and sequence level feature representations
    for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [ConveRTTokenizer]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow_text", "tensorflow_hub"]

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:

        super(ConveRTFeaturizer, self).__init__(component_config)

    def __get_signature(self, signature: Text, module: Any) -> NoReturn:
        """Retrieve a signature from a (hopefully loaded) TF model."""

        if not module:
            raise Exception(
                "ConveRTFeaturizer needs a proper loaded tensorflow module when used. "
                "Make sure to pass a module when training and using the component."
            )

        return module.signatures[signature]

    def _compute_features(
        self, batch_examples: List[Message], module: Any, attribute: Text = TEXT
    ) -> np.ndarray:

        sentence_encodings = self._compute_sentence_encodings(
            batch_examples, module, attribute
        )

        (
            sequence_encodings,
            number_of_tokens_in_sentence,
        ) = self._compute_sequence_encodings(batch_examples, module, attribute)

        return self._combine_encodings(
            sentence_encodings, sequence_encodings, number_of_tokens_in_sentence
        )

    def _compute_sentence_encodings(
        self, batch_examples: List[Message], module: Any, attribute: Text = TEXT
    ) -> np.ndarray:
        # Get text for attribute of each example
        batch_attribute_text = [ex.get(attribute) for ex in batch_examples]
        sentence_encodings = self._sentence_encoding_of_text(
            batch_attribute_text, module
        )

        # convert them to a sequence of 1
        return np.reshape(sentence_encodings, (len(batch_examples), 1, -1))

    def _compute_sequence_encodings(
        self, batch_examples: List[Message], module: Any, attribute: Text = TEXT
    ) -> Tuple[np.ndarray, List[int]]:
        list_of_tokens = [
            train_utils.tokens_without_cls(example, attribute)
            for example in batch_examples
        ]

        number_of_tokens_in_sentence = [
            len(sent_tokens) for sent_tokens in list_of_tokens
        ]

        # join the tokens to get a clean text to ensure the sequence length of
        # the returned embeddings from ConveRT matches the length of the tokens
        # (including sub-tokens)
        tokenized_texts = self._tokens_to_text(list_of_tokens)
        token_features = self._sequence_encoding_of_text(tokenized_texts, module)

        # ConveRT might split up tokens into sub-tokens
        # take the mean of the sub-token vectors and use that as the token vector
        token_features = train_utils.align_token_features(
            list_of_tokens, token_features
        )

        return token_features, number_of_tokens_in_sentence

    def _combine_encodings(
        self,
        sentence_encodings: np.ndarray,
        sequence_encodings: np.ndarray,
        number_of_tokens_in_sentence: List[int],
    ) -> np.ndarray:
        """Combine the sequence encodings with the sentence encodings.

        Append the sentence encoding to the end of the sequence encodings (position
        of CLS token)."""

        final_embeddings = []

        for index in range(len(number_of_tokens_in_sentence)):
            sequence_length = number_of_tokens_in_sentence[index]
            sequence_encoding = sequence_encodings[index][:sequence_length]
            sentence_encoding = sentence_encodings[index]

            # tile sequence encoding to duplicate as sentence encodings have size
            # 1024 and sequence encodings only have a dimensionality of 512
            sequence_encoding = np.tile(sequence_encoding, (1, 2))
            # add sentence encoding to the end (position of cls token)
            sequence_encoding = np.concatenate(
                [sequence_encoding, sentence_encoding], axis=0
            )

            final_embeddings.append(sequence_encoding)

        return np.array(final_embeddings)

    @staticmethod
    def _tokens_to_text(list_of_tokens: List[List[Token]]) -> List[Text]:
        """Convert list of tokens to text.

        Add a whitespace between two tokens if the end value of the first tokens is
        not the same as the end value of the second token."""

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

    def _sentence_encoding_of_text(self, batch: List[Text], module: Any) -> np.ndarray:
        signature = self.__get_signature("default", module)
        return signature(tf.convert_to_tensor(batch))["default"].numpy()

    def _sequence_encoding_of_text(self, batch: List[Text], module: Any) -> np.ndarray:
        signature = self.__get_signature("encode_sequence", module)

        return signature(tf.convert_to_tensor(batch))["sequence_encoding"].numpy()

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        *,
        tf_hub_module: Any = None,
        **kwargs: Any,
    ) -> None:

        if config is not None and config.language != "en":
            common_utils.raise_warning(
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

                batch_features = self._compute_features(
                    batch_examples, tf_hub_module, attribute
                )

                for index, ex in enumerate(batch_examples):
                    ex.set(
                        DENSE_FEATURE_NAMES[attribute],
                        self._combine_with_existing_dense_features(
                            ex, batch_features[index], DENSE_FEATURE_NAMES[attribute]
                        ),
                    )

    def process(
        self, message: Message, *, tf_hub_module: Any = None, **kwargs: Any
    ) -> None:
        features = self._compute_features([message], tf_hub_module)[0]
        message.set(
            DENSE_FEATURE_NAMES[TEXT],
            self._combine_with_existing_dense_features(
                message, features, DENSE_FEATURE_NAMES[TEXT]
            ),
        )
