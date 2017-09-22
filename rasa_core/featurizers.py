from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os

import jsonpickle
import numpy as np
from builtins import str

from rasa_core import utils

logger = logging.getLogger(__name__)


class Featurizer(object):
    """Transform the conversations state into machine learning formats.

    Featurizer decides how the bot will transform the conversation state to a
    format which a classifier can read."""

    def encode(self, active_features, input_feature_map):
        raise NotImplementedError("Featurizer must have the capacity to "
                                  "encode features to a vector")

    def decode(self, feature_vec, input_feature_map, ndigits=8):
        raise NotImplementedError("Featurizer must be able to "
                                  "decode features from a vector")

    def persist(self, path):
        featurizer_file = os.path.join(path, "featurizer.json")
        with io.open(featurizer_file, 'w') as f:
            f.write(str(jsonpickle.encode(self)))

    @staticmethod
    def load(path):
        featurizer_file = os.path.join(path, "featurizer.json")
        if os.path.isfile(featurizer_file):
            with io.open(featurizer_file, 'r') as f:
                _json = f.read()
            return jsonpickle.decode(_json)
        else:
            logger.info("Couldn't load featurizer for policy. "
                        "File '{}' doesn't exist. ".format(featurizer_file))
            return None


class BinaryFeaturizer(Featurizer):
    """Assumes all features are binary.

    All features should be either on or off, denoting them with 1 or 0."""

    def encode(self, active_features, input_feature_map):
        """Returns a binary vector indicating which features are active.

        Given a dictionary of active_features (e.g. 'intent_greet',
        'prev_action_listen',...) return a binary vector indicating which
        features of `self.input_features` are in the bag. NB it's a
        regular double precision float array type.

        For example with two active features out of five possible features
        this would return a vector like `[0 0 1 0 1]`

        If this is just a padding vector we set all values to `-1`.
        padding vectors are specified by a `None` or `[None]`
        value for active_features."""

        num_features = len(input_feature_map.keys())
        if active_features is None or None in active_features:
            return np.ones(num_features, dtype=np.int32) * -1
        else:
            # we are going to use floats and convert to int later if possible
            used_features = np.zeros(num_features, dtype=float)
            using_only_ints = True
            best_intent = None
            best_intent_prob = 0.0

            for feature_name, prob in active_features.items():
                if feature_name.startswith('intent_'):
                    if prob >= best_intent_prob:
                        best_intent = feature_name
                        best_intent_prob = prob
                elif feature_name in input_feature_map:
                    if prob != 0.0:
                        idx = input_feature_map[feature_name]
                        used_features[idx] = prob
                        using_only_ints = using_only_ints and utils.is_int(prob)
                else:
                    logger.debug(
                            "Feature '{}' (value: '{}') could not be found in "
                            "feature map. Make sure you added all intents and "
                            "entities to the domain".format(feature_name, prob))

            if best_intent is not None:
                # finding the maximum confidence intent and
                # appending it to the active_features val
                index_in_feature_list = input_feature_map.get(best_intent)
                if index_in_feature_list is not None:
                    used_features[index_in_feature_list] = 1
                else:
                    logger.warn(
                            "Couldn't set most probable feature '{}', "
                            "it wasn't found in the feature list of the domain."
                            " Make sure you added all intents and "
                            "entities to the domain.".format(best_intent))

            if using_only_ints:
                # this is an optimization - saves us a bit of memory
                return used_features.astype(np.int32)
            else:
                return used_features

    def decode(self, feature_vec, input_features, ndigits=8):
        """Reverse operation to binary_encoded_features

        :param feature_vec: binary feature vector
        :param input_features: list of all features
        :param ndigits: ignored
        :return: dictionary of active features
        """

        reversed_features = []
        for bf in feature_vec:
            if np.sum(np.where(bf == 1)) > 0:
                feature_tuples = []
                feat_names = list(np.array(input_features)[np.where(bf == 1)])
                for feat_name in feat_names:
                    feature_tuples.append((feat_name, 1))
                reversed_features.append(feature_tuples)
            else:
                reversed_features.append(None)
        return reversed_features


class ProbabilisticFeaturizer(Featurizer):
    """Uses intent probabilities of the NLU and feeds them into the model."""

    def encode(self, active_features, input_feature_map):
        """Returns a binary vector indicating active features,
        but with intent features given with a probability.

        Given a dictionary of active_features (e.g. 'intent_greet',
        'prev_action_listen',...) and intent probabilities
        from rasa_nlu, will a binary vector indicating which features of
        `self.input_features` are active.

        For example with two active features and an uncertain intent out of
        five possible features this would return a vector
        like `[0.3, 0.7, 1, 0, 1]`.

        If this is just a padding vector we set all values to `-1`.
        padding vectors are specified by a `None` or `[None]`
        value for active_features."""

        num_features = len(input_feature_map.keys())
        if active_features is None or None in active_features:
            return np.ones(num_features, dtype=np.int32) * -1
        else:

            intent_probs = {key: value
                            for key, value in active_features.items()
                            if key.startswith('intent_')}

            used_features = np.zeros(num_features, dtype=np.float)
            for active_feature, value in active_features.items():
                if active_feature in input_feature_map:
                    idx = input_feature_map[active_feature]
                    used_features[idx] = value
                else:
                    logger.debug(
                            "Found feature not in feature map. "
                            "Name: {} Value: {}".format(active_feature, value))
            for intent, probability in intent_probs.values():
                used_features[input_feature_map[intent]] = probability
            return used_features

    def decode(self, feature_vec, input_features, ndigits=8):
        """Reverse operation to binary_encoded_features

        :param feature_vec: binary feature vector
        :return: dictionary of active features, with their associated confidence
        """

        reversed_features = []
        for bf in feature_vec:
            if np.sum(np.where(bf > 0.)) > 0:
                active_features = np.argwhere(bf > 0.)
                feature_tuples = []
                for feature in active_features:
                    feat_name = list(input_features)[feature[0]]
                    if ndigits is not None:
                        feat_value = round(bf[feature[0]], ndigits)
                    else:
                        feat_value = bf[feature[0]]
                    feature_tuples.append((feat_name, feat_value))
                reversed_features.append(feature_tuples)
            else:
                reversed_features.append(None)
        return reversed_features
