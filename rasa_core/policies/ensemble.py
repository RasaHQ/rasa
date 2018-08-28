from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
from builtins import str
import typing
from typing import Text, Optional, Any, List, Dict

import rasa_core
from rasa_core import utils, training, constants
from rasa_core.events import SlotSet, ActionExecuted
from rasa_core.exceptions import UnsupportedDialogueModelError
from rasa_core.featurizers import MaxHistoryTrackerFeaturizer

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.policies.policy import Policy
    from rasa_core.trackers import DialogueStateTracker


class PolicyEnsemble(object):
    def __init__(self, policies, action_fingerprints=None):
        # type: (List[Policy], Optional[Dict]) -> None
        self.policies = policies
        self.training_trackers = None

        if action_fingerprints:
            self.action_fingerprints = action_fingerprints
        else:
            self.action_fingerprints = {}

    @staticmethod
    def _training_events_from_trackers(training_trackers):
        events_metadata = defaultdict(set)

        for t in training_trackers:
            tracker = t.init_copy()
            for event in t.events:
                tracker.update(event)
                if not isinstance(event, ActionExecuted):
                    action_name = tracker.latest_action_name
                    events_metadata[action_name].add(event)

        return events_metadata

    def train(self, training_trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, Any) -> None
        if training_trackers:
            for policy in self.policies:
                policy.train(training_trackers, domain, **kwargs)
            self.training_trackers = training_trackers
        else:
            logger.info("Skipped training, because there are no "
                        "training samples.")

    def probabilities_using_best_policy(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        raise NotImplementedError

    def _max_histories(self):
        # type: () -> List[Optional[int]]
        """Return max history."""

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
        if not training_events:
            return None

        action_fingerprints = {}
        for k, vs in training_events.items():
            slots = list({v.key for v in vs if isinstance(v, SlotSet)})
            action_fingerprints[k] = {"slots": slots}
        return action_fingerprints

    def _persist_metadata(self, path, dump_flattened_stories=False):
        # type: (Text, bool) -> None
        """Persists the domain specification to storage."""

        # make sure the directory we persist to exists
        domain_spec_path = os.path.join(path, 'policy_metadata.json')
        training_data_path = os.path.join(path, 'stories.md')
        utils.create_dir_for_file(domain_spec_path)

        policy_names = [utils.module_path_from_instance(p)
                        for p in self.policies]

        training_events = self._training_events_from_trackers(
                self.training_trackers)

        action_fingerprints = self._create_action_fingerprints(training_events)

        metadata = {
            "action_fingerprints": action_fingerprints,
            "rasa_core": rasa_core.__version__,
            "python": ".".join([str(s) for s in sys.version_info[:3]]),
            "max_histories": self._max_histories(),
            "ensemble_name": self.__module__ + "." + self.__class__.__name__,
            "policy_names": policy_names
        }

        utils.dump_obj_as_json_to_file(domain_spec_path, metadata)

        # if there are lots of stories, saving flattened stories takes a long
        # time, so this is turned off by default
        if dump_flattened_stories:
            training.persist_data(self.training_trackers, training_data_path)

    def persist(self, path, dump_flattened_stories=False):
        # type: (Text, bool) -> None
        """Persists the policy to storage."""

        self._persist_metadata(path, dump_flattened_stories)

        for i, policy in enumerate(self.policies):
            dir_name = 'policy_{}_{}'.format(i, type(policy).__name__)
            policy_path = os.path.join(path, dir_name)
            policy.persist(policy_path)

    @classmethod
    def load_metadata(cls, path):
        metadata_path = os.path.join(path, 'policy_metadata.json')
        with io.open(os.path.abspath(metadata_path)) as f:
            metadata = json.loads(f.read())
        return metadata

    @staticmethod
    def ensure_model_compatibility(metadata, version_to_check=None):
        from packaging import version

        if version_to_check is None:
            version_to_check = constants.MINIMUM_COMPATIBLE_VERSION

        model_version = metadata.get("rasa_core", "0.0.0")
        if version.parse(model_version) < version.parse(version_to_check):
            raise UnsupportedDialogueModelError(
                    "The model version is to old to be "
                    "loaded by this Rasa Core instance. "
                    "Either retrain the model, or run with"
                    "an older version. "
                    "Model version: {} Instance version: {} "
                    "Minimal compatible version: {}"
                    "".format(model_version, rasa_core.__version__,
                              version_to_check),
                    model_version)

    @classmethod
    def load(cls, path):
        # type: (Text) -> PolicyEnsemble
        """Loads policy and domain specification from storage"""

        metadata = cls.load_metadata(path)
        cls.ensure_model_compatibility(metadata)
        policies = []
        for i, policy_name in enumerate(metadata["policy_names"]):
            policy_cls = utils.class_from_module_path(policy_name)
            dir_name = 'policy_{}_{}'.format(i, policy_cls.__name__)
            policy_path = os.path.join(path, dir_name)
            policy = policy_cls.load(policy_path)
            policies.append(policy)
        ensemble_cls = utils.class_from_module_path(
                metadata["ensemble_name"])
        fingerprints = metadata.get("action_fingerprints", {})
        ensemble = ensemble_cls(policies, fingerprints)
        return ensemble

    def continue_training(self, trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, Any) -> None

        self.training_trackers.extend(trackers)
        for p in self.policies:
            p.continue_training(self.training_trackers, domain, **kwargs)


class SimplePolicyEnsemble(PolicyEnsemble):

    def probabilities_using_best_policy(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        result = None
        max_confidence = -1
        best_policy_name = None
        for i, p in enumerate(self.policies):
            probabilities = p.predict_action_probabilities(tracker, domain)
            confidence = np.max(probabilities)
            if confidence > max_confidence:
                max_confidence = confidence
                result = probabilities
                best_policy_name = 'policy_{}_{}'.format(i, type(p).__name__)
        # normalize probablilities
        if np.sum(result) != 0:
            result = result / np.linalg.norm(result)
        logger.debug("Predicted next action using {}"
                     "".format(best_policy_name))
        return result
