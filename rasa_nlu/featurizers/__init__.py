from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from rasa_nlu.components import Component


class Featurizer(Component):
    def _combine_with_existing_text_features(self, message, additional_features):
        import numpy as np

        if message.get("text_features") is not None:
            return np.hstack((message.get("text_features"), additional_features))
        else:
            return additional_features
