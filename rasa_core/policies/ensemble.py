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
from rasa_core.events import SlotSet
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.data import DialogueTrainingData

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer


class PolicyEnsemble(object):
    def __init__(self, policies, action_fingerprints=None):
        self.policies = policies
        self.training_metadata = {}

        if action_fingerprints:
            self.action_fingerprints = action_fingerprints
        else:
            self.action_fingerprints = {}

    def max_history(self):
        # type: () -> Optional[int]
        """Return max history, only works if the ensemble is already trained."""

        if self.policies:
            return self.policies[0].max_history
        else:
            return None

    def train(self, training_data, domain, featurizer, **kwargs):
        # type: (DialogueTrainingData, Domain, Featurizer, **Any) -> None
        if not training_data.is_empty():
            for policy in self.policies:
                policy.prepare(featurizer,
                               max_history=training_data.max_history())
                policy.train(training_data, domain, **kwargs)
                self.training_metadata.update(training_data.metadata)
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
        logger.debug("Predicted next action '{}' with prob {:.2f}.".format(
                domain.action_for_index(max_index).name(),
                probabilities[max_index]))
        return max_index

    def probabilities_using_best_policy(self, tracker, domain):
        raise NotImplementedError

    @staticmethod
    def _create_action_fingerprints(training_events):
        """Fingerprint each action using the events it created during train.

        This allows us to emit warnings when the model is used
        if an action does things it hasn't done during training."""

        action_fingerprints = {}
        for k, vs in training_events.items():
            slots = list({v.key for v in vs if isinstance(v, SlotSet)})
            action_fingerprints[k] = {"slots": slots}
        return action_fingerprints

    def _persist_metadata(self, path, max_history):
        # type: (Text, Optional[int]) -> None
        """Persists the domain specification to storage."""

        # make sure the directory we persist to exists
        domain_spec_path = os.path.join(path, 'policy_metadata.json')
        utils.create_dir_for_file(domain_spec_path)

        policy_names = [utils.module_path_from_instance(p)
                        for p in self.policies]
        training_events = self.training_metadata.get("events", {})
        action_fingerprints = self._create_action_fingerprints(training_events)

        metadata = {
            "action_fingerprints": action_fingerprints,
            "rasa_core": rasa_core.__version__,
            "max_history": max_history,
            "ensemble_name": self.__module__ + "." + self.__class__.__name__,
            "policy_names": policy_names
        }

        utils.dump_obj_as_json_to_file(domain_spec_path, metadata)

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""

        self._persist_metadata(path, self.max_history())

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
        fingerprints = metadata.get("action_fingerprints", {})
        ensemble = ensemble_cls(policies, fingerprints)
        return ensemble


class SimplePolicyEnsemble(PolicyEnsemble):
    def __init__(self, policies, known_slot_events=None):
        super(SimplePolicyEnsemble, self).__init__(policies, known_slot_events)

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
