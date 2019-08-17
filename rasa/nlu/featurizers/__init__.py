import numpy as np

from typing import Any
from rasa.nlu.training_data import Message
from rasa.nlu.components import Component


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_text_features(
        message: Message, additional_features: Any
    ) -> Any:
        if message.get("text_features") is not None:
            return np.hstack((message.get("text_features"), additional_features))
        else:
            return additional_features

    @staticmethod
    def _combine_with_existing_ner_features(
        message: Message, additional_features: Any
    ) -> Any:
        if message.get("ner_features") is not None:
            return np.concatenate(
                (message.get("ner_features"), additional_features), axis=1
            )
        else:
            return additional_features
