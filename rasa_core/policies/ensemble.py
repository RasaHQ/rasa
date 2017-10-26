from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os

import numpy as np
import typing
from builtins import str
from typing import Text, Optional

import rasa_core
from rasa_core import utils
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer


class PolicyEnsemble(object):
    def __init__(self, policies):
        self.policies = policies

    def train(self, X, y, domain, featurizer, **kwargs):
        if not utils.is_training_data_empty(X):
            for policy in self.policies:
                policy.prepare(featurizer, max_history=X.shape[1])
                policy.train(X, y, domain, **kwargs)
        else:
            logger.info("Skipped training, because there are no "
                        "training samples.")

    def predict_next_action(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> (float, int)
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use ML to
        predict the action. Returns the index of the next action"""
        probabilities = self.probabilities_using_best_policy(tracker, domain)
        max_index = np.argmax(probabilities)
        logger.debug("Predicted next action #{} with prob {:.2f}.".format(
                max_index, probabilities[max_index]))
        return max_index

    def probabilities_using_best_policy(self, tracker, domain):
        raise NotImplementedError

    def _persist_metadata(self, path, max_history):
        # type: (Text, Optional[int]) -> None
        """Persists the domain specification to storage."""

        domain_spec_path = os.path.join(path, 'policy_metadata.json')
        utils.create_dir_for_file(domain_spec_path)
        policy_names = [p.__module__ + "." + p.__class__.__name__
                        for p in self.policies]
        metadata = {
            "rasa_core": rasa_core.__version__,
            "max_history": max_history,
            "ensemble_name": self.__module__ + "." + self.__class__.__name__,
            "policy_names": policy_names
        }
        with io.open(domain_spec_path, 'w') as f:
            f.write(str(json.dumps(metadata, indent=2)))

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""

        if self.policies:
            self._persist_metadata(path, self.policies[0].max_history)
        else:
            self._persist_metadata(path, None)

        for policy in self.policies:
            policy.persist(path)

    @classmethod
    def load_metadata(cls, path):
        matadata_path = os.path.join(path, 'policy_metadata.json')
        with io.open(matadata_path) as f:
            metadata = json.loads(f.read())
        return metadata

    @classmethod
    def load(cls, path, featurizer):
        # type: (Text, Optional[Featurizer]) -> PolicyEnsemble
        """Loads policy and domain specification from storage"""

        metadata = cls.load_metadata(path)
        policies = []
        for policy_name in metadata["policy_names"]:
            policy_cls = utils.class_from_module_path(policy_name)
            policy = policy_cls.load(path, featurizer, metadata["max_history"])
            policies.append(policy)
        ensemble_cls = utils.class_from_module_path(metadata["ensemble_name"])
        ensemble = ensemble_cls(policies)
        return ensemble


class SimplePolicyEnsemble(PolicyEnsemble):
    def __init__(self, policies):
        super(SimplePolicyEnsemble, self).__init__(policies)

    def probabilities_using_best_policy(self, tracker, domain):
        result = None
        max_confidence = -1
        for p in self.policies:
            probabilities = p.predict_action_probabilities(tracker, domain)
            confidence = np.max(probabilities)
            if confidence > max_confidence:
                max_confidence = confidence
                result = probabilities
        return result
