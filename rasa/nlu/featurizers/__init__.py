import numpy as np

from rasa.nlu.components import Component


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_text_features(message, additional_features):
        if message.get("text_features") is not None:
            return np.hstack((message.get("text_features"), additional_features))
        else:
            return additional_features
