import numpy as np

from rasa.nlu.components import Component
from rasa.nlu.constants import MESSAGE_TEXT_ATTRIBUTE


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_features(
        message, additional_features, feature_name=MESSAGE_TEXT_ATTRIBUTE
    ):
        if message.get("{}_features".format(feature_name)) is not None:
            return np.hstack(
                (message.get("{}_features".format(feature_name)), additional_features)
            )
        else:
            return additional_features
