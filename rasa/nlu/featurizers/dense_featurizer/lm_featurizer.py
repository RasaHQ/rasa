import numpy as np
from typing import Any, Optional, Text, List

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TEXT,
    LANGUAGE_MODEL_DOCS,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    SEQUENCE_FEATURES,
    SENTENCE_FEATURES,
)


class LanguageModelFeaturizer(DenseFeaturizer):
    """Featurizer using transformer based language models.

        Uses the output of HFTransformersNLP component to set the sequence and sentence
        level representations for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Any]:
        return [HFTransformersNLP, LanguageModelTokenizer]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        for example in training_data.training_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_lm_features(example, attribute)

    def get_doc(self, message: Message, attribute: Text) -> Any:

        return message.get(LANGUAGE_MODEL_DOCS[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:

        self._set_lm_features(message)

    def _set_lm_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Adds the precomputed word vectors to the messages features."""

        doc = self.get_doc(message, attribute)

        if doc is not None:
            sequence_features = doc[SEQUENCE_FEATURES]
            sentence_features = doc[SENTENCE_FEATURES]

            features = np.concatenate([sequence_features, sentence_features])

            features = self._combine_with_existing_dense_features(
                message, features, DENSE_FEATURE_NAMES[attribute]
            )
            message.set(DENSE_FEATURE_NAMES[attribute], features)
