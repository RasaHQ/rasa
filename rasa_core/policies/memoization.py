from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
import io
import json
import logging
import os
import zlib

import numpy as np
from builtins import str, bytes
from numpy import ndarray
from rasa_core.trackers import DialogueStateTracker
from typing import Optional, List, Any

from rasa_core.policies.policy import Policy
from rasa_core import utils

logger = logging.getLogger(__name__)

ENABLE_FEATURE_STRING_COMPRESSION = True


class MemoizationPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    def __init__(self, lookup=None, featurizer=None, max_history=None):
        self.lookup = lookup if lookup is not None else {}
        self.is_enabled = True
        super(MemoizationPolicy, self).__init__(featurizer, max_history)

    def toggle(self, activate):
        self.is_enabled = activate

    def memorise(self, X, y, domain):
        assert X.shape[1] == self.max_history, \
            ("Trying to mem featurized data with {} historic turns. "
             "Expected {}".format(X.shape[1], self.max_history))
        self.lookup = {}
        self.add(X, y, domain)

    def _create_partial_histories(self, x):
        augmented = [np.array(x)]
        original_x = np.array(x)
        for i in range(0, self.max_history - 1):
            original_x[i, :] = -1
            augmented.append(np.array(original_x))
        return augmented

    def add(self, X, y, domain):
        assert X.shape[1] == self.max_history, \
            ("Trying to mem featurized data with {} historic turns. "
             "Expected {}".format(X.shape[1], self.max_history))
        for _x, _y in zip(X, y):
            for _x_augmented in self._create_partial_histories(_x):
                feature_key = self._feature_vector_to_str(_x_augmented, domain)
                self.lookup[feature_key] = _y.item()

    def _feature_vector_to_str(self, x, domain):
        decoded_features = self.featurizer.decode(x,
                                                  domain.input_features,
                                                  ndigits=8)
        feature_str = json.dumps(decoded_features).replace("\"", "")
        if ENABLE_FEATURE_STRING_COMPRESSION:
            compressed = zlib.compress(bytes(feature_str, "utf-8"))
            return base64.b64encode(compressed).decode("utf-8")
        else:
            return feature_str

    def recall(self, x, domain):
        if x.ndim == 3:
            # remove the batch dimension
            x = np.squeeze(x, axis=(0,))
        return self.lookup.get(self._feature_vector_to_str(x, domain))

    def train(self, X, y, domain, **kwargs):
        # type: (ndarray, List[int], Domain, **Any) -> None
        """Trains the policy on given training data."""

        self.memorise(X, y, domain)

    def continue_training(self, X, y, domain, **kwargs):
        # fit to one extra example
        self.add(X, y, domain)

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> (float, Optional[int])
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use ML to
        predict the action. Returns the index of the next action"""
        x = self.featurize(tracker, domain)
        tracker_state = ["{}".format(e)
                         for e in self.featurizer.decode(x,
                                                         domain.input_features)]
        logger.debug('Current tracker state [\n\t{}]'.format(
                "\n\t".join(tracker_state)))

        memorised = self.recall(x, domain)
        result = [0.0] * domain.num_actions
        if memorised is not None and self.is_enabled:
            logger.debug("Used memorised next action '{}'".format(memorised))
            result[memorised] = 1.0
        return result

    def persist(self, path):
        memorized_file = os.path.join(path, 'memorized_turns.json')
        data = {
            "lookup": self.lookup
        }
        utils.create_dir_for_file(memorized_file)
        with io.open(memorized_file, 'w') as f:
            f.write(str(json.dumps(data, indent=2)))

    @classmethod
    def load(cls, path, featurizer, max_history):
        memorized_file = os.path.join(path, 'memorized_turns.json')
        if os.path.isfile(memorized_file):
            with io.open(memorized_file) as f:
                data = json.loads(f.read())
            return cls(data["lookup"],
                       featurizer=featurizer,
                       max_history=max_history)
        else:
            logger.info("Couldn't load memoization for policy. "
                        "File '{}' doesn't exist. Falling back to empty "
                        "turn memory.".format(memorized_file))
            return None
