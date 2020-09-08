from typing import Any, Optional, Text, List, Type

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    LANGUAGE_MODEL_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    SEQUENCE_FEATURES,
    SENTENCE_FEATURES,
    FEATURIZER_CLASS_ALIAS,
)
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE


class LanguageModelFeaturizer(DenseFeaturizer):
    """Featurizer using transformer based language models.

    Uses the output of HFTransformersNLP component to set the sequence and sentence
    level representations for dense featurizable attributes of each message object.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
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

    def _get_doc(self, message: Message, attribute: Text) -> Any:
        """
        Get the language model doc. A doc consists of
        {'token_ids': ..., 'tokens': ...,
        'sequence_features': ..., 'sentence_features': ...}
        """
        return message.get(LANGUAGE_MODEL_DOCS[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        """Sets the dense features from the language model doc to the incoming
        message."""
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            self._set_lm_features(message, attribute)

    def _set_lm_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Adds the precomputed word vectors to the messages features."""
        doc = self._get_doc(message, attribute)

        if doc is None:
            return

        sequence_features = doc[SEQUENCE_FEATURES]
        sentence_features = doc[SENTENCE_FEATURES]

        final_sequence_features = Features(
            sequence_features,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            sentence_features,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.component_config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)
