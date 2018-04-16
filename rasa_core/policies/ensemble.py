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
from typing import Text, Optional, Any, List, Dict

import rasa_core
from rasa_core import utils
from rasa_core.events import SlotSet
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.featurizers import Featurizer
    from rasa_core.policies.policy import Policy
    from rasa_core.trackers import DialogueStateTracker


class PolicyEnsemble(object):
    def __init__(self, policies, action_fingerprints=None):
        # type: (List[Policy], Optional[Dict]) -> None
        self.policies = policies
        self.training_events = None

        if action_fingerprints:
            self.action_fingerprints = action_fingerprints
        else:
            self.action_fingerprints = {}

    def train(self, training_trackers, training_events, domain, **kwargs):
        # type: (List[DialogueStateTracker], Dict, Domain, **Any) -> None
        if training_trackers:
            for policy in self.policies:
                policy.train(training_trackers, domain, **kwargs)
            self.training_events = training_events
        else:
            logger.info("Skipped training, because there are no "
                        "training samples.")

    def probabilities_using_best_policy(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        raise NotImplementedError

    def predict_next_action(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> int
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use ML to
        predict the action. Returns the index of the next action"""
        probabilities = self.probabilities_using_best_policy(tracker, domain)
        max_index = int(np.argmax(probabilities))
        logger.debug("Predicted next action '{}' with prob {:.2f}.".format(
                domain.action_for_index(max_index).name(),
                probabilities[max_index]))
        return max_index

    def _max_histories(self):
        # type: () -> List[Optional[int]]
        """Return max history, only works if the ensemble is already trained."""

        max_histories = []
        for p in self.policies:
            if isinstance(p.featurizer, MaxHistoryTrackerFeaturizer):
                max_histories.append(p.featurizer.max_history)
            else:
                max_histories.append(None)
        return max_histories

    @staticmethod
    def _create_action_fingerprints(training_events):
        """Fingerprint each action using the events it created during train.

        This allows us to emit warnings when the model is used
        if an action does things it hasn't done during training."""
        if training_events:
            action_fingerprints = {}
            for k, vs in training_events.items():
                slots = list({v.key for v in vs if isinstance(v, SlotSet)})
                action_fingerprints[k] = {"slots": slots}
            return action_fingerprints
        else:
            return None

    def _persist_metadata(self, path, max_histories):
        # type: (Text, List[Optional[int]]) -> None
        """Persists the domain specification to storage."""

        # make sure the directory we persist to exists
        domain_spec_path = os.path.join(path, 'policy_metadata.json')
        utils.create_dir_for_file(domain_spec_path)

        policy_names = [utils.module_path_from_instance(p)
                        for p in self.policies]
        action_fingerprints = self._create_action_fingerprints(self.training_events)

        metadata = {
            "action_fingerprints": action_fingerprints,
            "rasa_core": rasa_core.__version__,
            "max_histories": max_histories,
            "ensemble_name": self.__module__ + "." + self.__class__.__name__,
            "policy_names": policy_names
        }

        utils.dump_obj_as_json_to_file(domain_spec_path, metadata)

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""

        self._persist_metadata(path, self._max_histories())

        for i, policy in enumerate(self.policies):
            # TODO better way then many folders?
            # TODO delete old files from these folders
            policy_path = os.path.join(path, 'policy_{}'.format(i))
            policy.persist(policy_path)

    @classmethod
    def load_metadata(cls, path):
        matadata_path = os.path.join(path, 'policy_metadata.json')
        with io.open(matadata_path) as f:
            metadata = json.loads(f.read())
        return metadata

    @classmethod
    def load(cls, path):
        # type: (Text) -> PolicyEnsemble
        """Loads policy and domain specification from storage"""

        metadata = cls.load_metadata(path)
        policies = []
        for i, policy_name in enumerate(metadata["policy_names"]):
            policy_cls = utils.class_from_module_path(policy_name)
            policy_path = os.path.join(path, 'policy_{}'.format(i))
            policy = policy_cls.load(policy_path)
            policies.append(policy)
        ensemble_cls = utils.class_from_module_path(metadata["ensemble_name"])
        fingerprints = metadata.get("action_fingerprints", {})
        ensemble = ensemble_cls(policies, fingerprints)
        return ensemble


class SimplePolicyEnsemble(PolicyEnsemble):
    def __init__(self, policies, known_slot_events=None):
        # TODO is known_slot_events is the same as action_fingerprints?
        # TODO if so remove this init, else here is an error
        super(SimplePolicyEnsemble, self).__init__(policies, known_slot_events)

    def probabilities_using_best_policy(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        result = None
        max_confidence = -1
        for p in self.policies:
            probabilities = p.predict_action_probabilities(tracker, domain)
            confidence = np.max(probabilities)
            if confidence > max_confidence:
                max_confidence = confidence
                result = probabilities
        return result
