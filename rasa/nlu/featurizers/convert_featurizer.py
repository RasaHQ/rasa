import logging
import os
import re
import scipy.sparse
from rasa.nlu.featurizers import Featurizer
from typing import Any, Dict, List, Optional, Text
from rasa.nlu import utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ATTRIBUTES,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_VECTOR_FEATURE_NAMES,
    SPACY_FEATURIZABLE_ATTRIBUTES,
)
import numpy as np
import tensorflow_text
import tensorflow_hub as tfhub

logger = logging.getLogger(__name__)


class ConvertFeaturizer(Featurizer):

    provides = [
        MESSAGE_VECTOR_FEATURE_NAMES[attribute]
        for attribute in SPACY_FEATURIZABLE_ATTRIBUTES
    ]

    def _load_model(self):

        model_url = "http://models.poly-ai.com/convert/v1/model.tar.gz"

        self.module = tfhub.Module(model_url)

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:

        super(ConvertFeaturizer, self).__init__(component_config)

        self._load_model()

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig],
        **kwargs: Any,
    ) -> None:

        for example in training_data.intent_examples:
            for attribute in SPACY_FEATURIZABLE_ATTRIBUTES:
                self._set_lm_features(example, attribute)

    def _set_lm_features(self, example, attribute=MESSAGE_TEXT_ATTRIBUTE):

        message_attribute_text = example.get(attribute)
        if message_attribute_text:
            # Encode text
            features = self.module([message_attribute_text])[0]
            features = self._combine_with_existing_features(
                example, features, MESSAGE_VECTOR_FEATURE_NAMES[attribute]
            )
            example.set(MESSAGE_VECTOR_FEATURE_NAMES[attribute], features)

    def process(self, message: Message, **kwargs: Any) -> None:

        self._set_lm_features(message)
