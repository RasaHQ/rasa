import logging
import warnings
from rasa.nlu.featurizers.featurizer import Featurizer
from typing import Any, Dict, List, Optional, Text, Tuple
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TEXT_ATTRIBUTE,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class ConveRTFeaturizer(Featurizer):

    provides = [
        DENSE_FEATURE_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {
        # if True return a sequence of features (return vector has size
        # token-size x feature-dimension)
        # if False token-size will be equal to 1
        "return_sequence": False
    }

    def _load_model(self) -> None:

        # needed in order to load model
        import tensorflow_text
        import tensorflow_hub as tfhub

        self.graph = tf.Graph()
        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"

        with self.graph.as_default():
            self.session = tf.Session()
            self.module = tfhub.Module(model_url)

            self.text_placeholder = tf.placeholder(dtype=tf.string, shape=[None])
            self.encoding_tensor = self.module(self.text_placeholder)
            self.session.run(tf.tables_initializer())
            self.session.run(tf.global_variables_initializer())

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:

        super(ConveRTFeaturizer, self).__init__(component_config)

        self._load_model()

        self.return_sequence = self.component_config["return_sequence"]

        if self.return_sequence:
            raise NotImplementedError(
                f"ConveRTFeaturizer always returns a feature vector of size "
                f"(1 x feature-dimensions). It cannot return a proper sequence "
                f"right now. ConveRTFeaturizer can only be used "
                f"with 'return_sequence' set to False. Also, any other featurizer "
                f"used next to ConveRTFeaturizer should have the flag "
                f"'return_sequence' set to False."
            )

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow_text", "tensorflow_hub"]

    def _compute_features(
        self, batch_examples: List[Message], attribute: Text = TEXT_ATTRIBUTE
    ) -> np.ndarray:

        # Get text for attribute of each example
        batch_attribute_text = [ex.get(attribute) for ex in batch_examples]

        batch_features = self._run_model_on_text(batch_attribute_text)

        return batch_features

    def _run_model_on_text(self, batch: List[Text]) -> np.ndarray:

        return self.session.run(
            self.encoding_tensor, feed_dict={self.text_placeholder: batch}
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        if config is not None and config.language != "en":
            warnings.warn(
                f"Since ``ConveRT`` model is trained only on an english "
                f"corpus of conversations, this featurizer should only be "
                f"used if your training data is in english language. "
                f"However, you are training in '{config.language}'."
            )

        batch_size = 64

        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:

            non_empty_examples = list(
                filter(lambda x: x.get(attribute), training_data.training_examples)
            )

            batch_start_index = 0

            while batch_start_index < len(non_empty_examples):

                batch_end_index = min(
                    batch_start_index + batch_size, len(non_empty_examples)
                )

                # Collect batch examples
                batch_examples = non_empty_examples[batch_start_index:batch_end_index]

                batch_features = self._compute_features(batch_examples, attribute)

                for index, ex in enumerate(batch_examples):

                    ex.set(
                        DENSE_FEATURE_NAMES[attribute],
                        self._combine_with_existing_dense_features(
                            ex,
                            np.expand_dims(batch_features[index], axis=0),
                            DENSE_FEATURE_NAMES[attribute],
                        ),
                    )

                batch_start_index += batch_size

    def process(self, message: Message, **kwargs: Any) -> None:

        feats = self._compute_features([message])[0]
        message.set(
            DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE],
            self._combine_with_existing_dense_features(
                message,
                np.expand_dims(feats, axis=0),
                DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE],
            ),
        )
