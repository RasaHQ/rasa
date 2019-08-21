import numpy as np

from rasa.nlu.components import Component


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_features(message, additional_features, attribute="text"):
        if message.get("{0}_features".format(attribute)) is not None:
            return np.hstack(
                (message.get("{0}_features".format(attribute)), additional_features)
            )
        else:
            return additional_features
