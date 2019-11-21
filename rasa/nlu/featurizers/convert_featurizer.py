import logging
from rasa.nlu.featurizers import Featurizer
from typing import Any, Dict, List, Optional, Text, Tuple
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_FEATURE_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class ConveRTFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_FEATURE_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    def _load_model(self) -> None:

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

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        super(ConveRTFeaturizer, self).__init__(component_config)

        self._load_model()

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow_text", "tensorflow_hub"]

    def _compute_features(
        self, batch_examples: List[Message], attribute: Text = MESSAGE_TEXT_ATTRIBUTE
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

        batch_size = 64

        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:

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
                        MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                        self._combine_with_existing_features(
                            ex,
                            batch_features[index],
                            MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                        ),
                    )

                batch_start_index += batch_size

    def process(self, message: Message, **kwargs: Any) -> None:

        feats = self._compute_features([message])[0]
        message.set(
            MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self._combine_with_existing_features(
                message, feats, MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            ),
        )
