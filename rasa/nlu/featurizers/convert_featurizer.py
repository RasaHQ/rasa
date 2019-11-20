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


class ConvertFeaturizer(Featurizer):

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

        super(ConvertFeaturizer, self).__init__(component_config)

        self._load_model()

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow", "tensorflow_text", "tensorflow_hub"]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        batch_size = 64

        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:

            batch_start_index = 0

            while batch_start_index < len(training_data.training_examples):

                batch_end_index = min(
                    batch_start_index + batch_size, len(training_data.training_examples)
                )

                batch_examples = training_data.training_examples[
                    batch_start_index:batch_end_index
                ]

                batch_feats = self._compute_features(batch_examples, attribute)

                for index, ex in enumerate(batch_examples):

                    # Don't set None features
                    if batch_feats[index] is not None:

                        ex.set(
                            MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                            self._combine_with_existing_features(
                                ex,
                                batch_feats[index],
                                MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                            ),
                        )

                batch_start_index += batch_size

    def _split_content_nocontent_examples(
        self, batch_attribute_text: List[Any]
    ) -> Tuple[List[Tuple[int, Any]], List[Tuple[int, Any]]]:

        # [(int, Text)]
        content_bearing_samples = [
            (index, example_text)
            for index, example_text in enumerate(batch_attribute_text)
            if example_text
        ]

        # [(int, None)]
        nocontent_bearing_samples = [
            (index, example_text)
            for index, example_text in enumerate(batch_attribute_text)
            if not example_text
        ]

        return content_bearing_samples, nocontent_bearing_samples

    def _compute_features(
        self, batch_examples: List[Message], attribute: Text = MESSAGE_TEXT_ATTRIBUTE
    ) -> List[np.ndarray]:

        # Get text for attribute of each example
        batch_attribute_text = [ex.get(attribute) for ex in batch_examples]

        # Split examples which have the attribute set from those which do not have it
        (
            content_bearing_examples,
            nocontent_bearing_examples,
        ) = self._split_content_nocontent_examples(batch_attribute_text)
        content_bearing_indices = [index for (index, _) in content_bearing_examples]

        # prepare the input for model
        text_for_model = [text for (index, text) in content_bearing_examples]

        # Get the features
        model_features = self._run_model_on_text(text_for_model)

        # Combine back index with features
        content_bearing_features = [
            (index, feature_vec)
            for (index, feature_vec) in zip(content_bearing_indices, model_features)
        ]

        # Combine all features
        batch_features = sorted(
            content_bearing_features + nocontent_bearing_examples, key=lambda x: x[0]
        )

        return [feature_vec for (_, feature_vec) in batch_features]

    def _run_model_on_text(self, batch: List[Text]) -> np.ndarray:

        return self.session.run(
            self.encoding_tensor, feed_dict={self.text_placeholder: batch}
        )

    def process(self, message: Message, **kwargs: Any) -> None:

        feats = self._compute_features([message])[0]
        message.set(
            MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self._combine_with_existing_features(
                message, feats, MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            ),
        )
