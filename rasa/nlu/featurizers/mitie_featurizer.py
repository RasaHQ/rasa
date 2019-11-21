import numpy as np
import typing
from typing import Any, List, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.tokenizers import Token
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    import mitie

from rasa.nlu.constants import (
    MESSAGE_RESPONSE_ATTRIBUTE,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_SPACY_FEATURES_NAMES,
    MESSAGE_VECTOR_FEATURE_NAMES,
)


class MitieFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_FEATURE_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES
    ]

    requires = [MESSAGE_TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES] + [
        "mitie_feature_extractor"
    ]

    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["mitie", "numpy"]

    def ndim(self, feature_extractor: "mitie.total_word_feature_extractor"):

        return feature_extractor.num_dimensions

    def get_tokens_by_attribute(self, example, attribute):

        return example.get(MESSAGE_TOKENS_NAMES[attribute])

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for example in training_data.intent_examples:

            for attribute in MESSAGE_ATTRIBUTES:

                attribute_tokens = self.get_tokens_by_attribute(example, attribute)
                if attribute_tokens is not None:

                    features = self.features_for_tokens(
                        attribute_tokens, mitie_feature_extractor
                    )
                    example.set(
                        MESSAGE_VECTOR_FEATURE_NAMES[attribute],
                        self._combine_with_existing_features(
                            example, features, MESSAGE_VECTOR_FEATURE_NAMES[attribute]
                        ),
                    )

    def process(self, message: Message, **kwargs: Any) -> None:

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        features = self.features_for_tokens(
            message.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE]),
            mitie_feature_extractor,
        )
        message.set(
            MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
            self._combine_with_existing_features(
                message, features, MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            ),
        )

    def _mitie_feature_extractor(self, **kwargs):
        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception(
                "Failed to train 'MitieFeaturizer'. "
                "Missing a proper MITIE feature extractor. "
                "Make sure this component is preceded by "
                "the 'MitieNLP' component in the pipeline "
                "configuration."
            )
        return mitie_feature_extractor

    def features_for_tokens(
        self,
        tokens: List[Token],
        feature_extractor: "mitie.total_word_feature_extractor",
    ) -> np.ndarray:

        vec = np.zeros(self.ndim(feature_extractor))
        for token in tokens:
            vec += feature_extractor.get_feature_vector(token.text)
        if tokens:
            return vec / len(tokens)
        else:
            return vec
