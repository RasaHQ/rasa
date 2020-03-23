import importlib
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Text, Optional, Any, List, Dict, Tuple, Set, NamedTuple

import rasa.core
import rasa.utils.io
from rasa.constants import MINIMUM_COMPATIBLE_VERSION, DOCS_BASE_URL, DOCS_URL_POLICIES

from rasa.core import utils
from rasa.core.constants import USER_INTENT_BACK, USER_INTENT_RESTART
from rasa.core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_BACK_NAME,
    ACTION_RESTART_NAME,
)
from rasa.core.domain import Domain
from rasa.core.events import SlotSet, ActionExecuted, ActionExecutionRejected, Event
from rasa.core.exceptions import UnsupportedDialogueModelError
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.policies.policy import Policy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.core import registry
from rasa.utils.common import class_from_module_path, raise_warning

logger = logging.getLogger(__name__)


class PolicyEnsemble:
    versioned_packages = ["rasa", "tensorflow", "sklearn"]

    def __init__(
        self, policies: List[Policy], action_fingerprints: Optional[Dict] = None
    ) -> None:
        self.policies = policies
        self.date_trained = None

        if action_fingerprints:
            self.action_fingerprints = action_fingerprints
        else:
            self.action_fingerprints = {}

        self._check_priorities()
        self._check_for_important_policies()

    def _check_for_important_policies(self) -> None:
        from rasa.core.policies.mapping_policy import MappingPolicy

        if not any(isinstance(policy, MappingPolicy) for policy in self.policies):
            logger.info(
                "MappingPolicy not included in policy ensemble. Default intents "
                "'{} and {} will not trigger actions '{}' and '{}'."
                "".format(
                    USER_INTENT_RESTART,
                    USER_INTENT_BACK,
                    ACTION_RESTART_NAME,
                    ACTION_BACK_NAME,
                )
            )

    @staticmethod
    def _training_events_from_trackers(training_trackers) -> Dict[Text, Set[Event]]:
        events_metadata = defaultdict(set)

        for t in training_trackers:
            tracker = t.init_copy()
            for event in t.events:
                tracker.update(event)
                if not isinstance(event, ActionExecuted):
                    action_name = tracker.latest_action_name
                    events_metadata[action_name].add(event)

        return events_metadata

    @staticmethod
    def check_domain_ensemble_compatibility(
        ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        """Check for elements that only work with certain policy/domain combinations."""

        from rasa.core.policies.form_policy import FormPolicy
        from rasa.core.policies.mapping_policy import MappingPolicy
        from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy

        policies_needing_validation = [
            FormPolicy,
            MappingPolicy,
            TwoStageFallbackPolicy,
        ]
        for policy in policies_needing_validation:
            policy.validate_against_domain(ensemble, domain)

    def _check_priorities(self) -> None:
        """Checks for duplicate policy priorities within PolicyEnsemble."""

        priority_dict = defaultdict(list)
        for p in self.policies:
            priority_dict[p.priority].append(type(p).__name__)

        for k, v in priority_dict.items():
            if len(v) > 1:
                raise_warning(
                    f"Found policies {v} with same priority {k} "
                    f"in PolicyEnsemble. When personalizing "
                    f"priorities, be sure to give all policies "
                    f"different priorities.",
                    docs=DOCS_URL_POLICIES,
                )

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        if training_trackers:
            for policy in self.policies:
                policy.train(training_trackers, domain, **kwargs)

            training_events = self._training_events_from_trackers(training_trackers)
            self.action_fingerprints = self._create_action_fingerprints(training_events)
        else:
            logger.info("Skipped training, because there are no training samples.")
        self.date_trained = datetime.now().strftime("%Y%m%d-%H%M%S")

    def probabilities_using_best_policy(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[Optional[List[float]], Optional[Text]]:
        raise NotImplementedError

    def _max_histories(self) -> List[Optional[int]]:
        """Return max history."""

        max_histories = []
        for p in self.policies:
            if isinstance(p.featurizer, MaxHistoryTrackerFeaturizer):
                max_histories.append(p.featurizer.max_history)
            else:
                max_histories.append(None)
        return max_histories

    @staticmethod
    def _create_action_fingerprints(
        training_events: Dict[Text, Set[Event]]
    ) -> Optional[Dict[Any, Dict[Text, List]]]:
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

    def _add_package_version_info(self, metadata: Dict[Text, Any]) -> None:
        """Adds version info for self.versioned_packages to metadata."""

        for package_name in self.versioned_packages:
            try:
                p = importlib.import_module(package_name)
                v = p.__version__  # pytype: disable=attribute-error
                metadata[package_name] = v
            except ImportError:
                pass

    def _persist_metadata(self, path: Text) -> None:
        """Persists the domain specification to storage."""

        # make sure the directory we persist exists
        domain_spec_path = os.path.join(path, "metadata.json")
        rasa.utils.io.create_directory_for_file(domain_spec_path)

        policy_names = [utils.module_path_from_instance(p) for p in self.policies]

        metadata = {
            "action_fingerprints": self.action_fingerprints,
            "python": ".".join([str(s) for s in sys.version_info[:3]]),
            "max_histories": self._max_histories(),
            "ensemble_name": self.__module__ + "." + self.__class__.__name__,
            "policy_names": policy_names,
            "trained_at": self.date_trained,
        }

        self._add_package_version_info(metadata)

        rasa.utils.io.dump_obj_as_json_to_file(domain_spec_path, metadata)

    def persist(self, path: Text) -> None:
        """Persists the policy to storage."""

        self._persist_metadata(path)

        for i, policy in enumerate(self.policies):
            dir_name = "policy_{}_{}".format(i, type(policy).__name__)
            policy_path = os.path.join(path, dir_name)
            policy.persist(policy_path)

    @classmethod
    def load_metadata(cls, path) -> Any:
        metadata_path = os.path.join(path, "metadata.json")
        metadata = json.loads(rasa.utils.io.read_file(os.path.abspath(metadata_path)))
        return metadata

    @staticmethod
    def ensure_model_compatibility(metadata, version_to_check=None) -> None:
        from packaging import version

        if version_to_check is None:
            version_to_check = MINIMUM_COMPATIBLE_VERSION

        model_version = metadata.get("rasa", "0.0.0")
        if version.parse(model_version) < version.parse(version_to_check):
            raise UnsupportedDialogueModelError(
                "The model version is too old to be "
                "loaded by this Rasa Core instance. "
                "Either retrain the model, or run with "
                "an older version. "
                "Model version: {} Instance version: {} "
                "Minimal compatible version: {}"
                "".format(model_version, rasa.__version__, version_to_check),
                model_version,
            )

    @classmethod
    def _ensure_loaded_policy(cls, policy, policy_cls, policy_name: Text):
        if policy is None:
            raise Exception(f"Failed to load policy {policy_name}: load returned None")
        elif not isinstance(policy, policy_cls):
            raise Exception(
                "Failed to load policy {}: "
                "load returned object that is not instance of its own class"
                "".format(policy_name)
            )

    @classmethod
    def load(cls, path: Text) -> "PolicyEnsemble":
        """Loads policy and domain specification from storage"""

        metadata = cls.load_metadata(path)
        cls.ensure_model_compatibility(metadata)
        policies = []
        for i, policy_name in enumerate(metadata["policy_names"]):
            policy_cls = registry.policy_from_module_path(policy_name)
            dir_name = f"policy_{i}_{policy_cls.__name__}"
            policy_path = os.path.join(path, dir_name)
            policy = policy_cls.load(policy_path)
            cls._ensure_loaded_policy(policy, policy_cls, policy_name)
            policies.append(policy)
        ensemble_cls = class_from_module_path(metadata["ensemble_name"])
        fingerprints = metadata.get("action_fingerprints", {})
        ensemble = ensemble_cls(policies, fingerprints)
        return ensemble

    @classmethod
    def from_dict(cls, policy_configuration: Dict[Text, Any]) -> List[Policy]:
        import copy

        policies = policy_configuration.get("policies") or policy_configuration.get(
            "policy"
        )
        if policies is None:
            raise InvalidPolicyConfig(
                "You didn't define any policies. "
                "Please define them under 'policies:' "
                "in your policy configuration file."
            )
        if len(policies) == 0:
            raise InvalidPolicyConfig(
                "The policy configuration file has to include at least one policy."
            )

        policies = copy.deepcopy(policies)  # don't manipulate passed `Dict`
        parsed_policies = []

        for policy in policies:
            policy_name = policy.pop("name")
            if policy.get("featurizer"):
                featurizer_func, featurizer_config = cls.get_featurizer_from_dict(
                    policy
                )

                if featurizer_config.get("state_featurizer"):
                    (
                        state_featurizer_func,
                        state_featurizer_config,
                    ) = cls.get_state_featurizer_from_dict(featurizer_config)

                    # override featurizer's state_featurizer
                    # with real state_featurizer class
                    featurizer_config["state_featurizer"] = state_featurizer_func(
                        **state_featurizer_config
                    )

                # override policy's featurizer with real featurizer class
                policy["featurizer"] = featurizer_func(**featurizer_config)

            try:
                constr_func = registry.policy_from_module_path(policy_name)
                try:
                    policy_object = constr_func(**policy)
                except TypeError as e:
                    raise Exception(f"Could not initialize {policy_name}. {e}")
                parsed_policies.append(policy_object)
            except (ImportError, AttributeError):
                raise InvalidPolicyConfig(
                    "Module for policy '{}' could not "
                    "be loaded. Please make sure the "
                    "name is a valid policy."
                    "".format(policy_name)
                )

        return parsed_policies

    @classmethod
    def get_featurizer_from_dict(cls, policy) -> Tuple[Any, Any]:
        # policy can have only 1 featurizer
        if len(policy["featurizer"]) > 1:
            raise InvalidPolicyConfig("policy can have only 1 featurizer")
        featurizer_config = policy["featurizer"][0]
        featurizer_name = featurizer_config.pop("name")
        featurizer_func = registry.featurizer_from_module_path(featurizer_name)

        return featurizer_func, featurizer_config

    @classmethod
    def get_state_featurizer_from_dict(cls, featurizer_config) -> Tuple[Any, Any]:
        # featurizer can have only 1 state featurizer
        if len(featurizer_config["state_featurizer"]) > 1:
            raise InvalidPolicyConfig("featurizer can have only 1 state featurizer")
        state_featurizer_config = featurizer_config["state_featurizer"][0]
        state_featurizer_name = state_featurizer_config.pop("name")
        state_featurizer_func = registry.featurizer_from_module_path(
            state_featurizer_name
        )

        return state_featurizer_func, state_featurizer_config


class Prediction(NamedTuple):
    """Stores the probabilities and the priority of the prediction."""

    probabilities: List[float]
    priority: int


class SimplePolicyEnsemble(PolicyEnsemble):
    @staticmethod
    def is_not_memo_policy(
        policy_name: Text, max_confidence: Optional[float] = None
    ) -> bool:
        is_memo = policy_name.endswith("_" + MemoizationPolicy.__name__)
        is_augmented = policy_name.endswith("_" + AugmentedMemoizationPolicy.__name__)
        # also check if confidence is 0, than it cannot be count as prediction
        return not (is_memo or is_augmented) or max_confidence == 0.0

    @staticmethod
    def _is_not_mapping_policy(
        policy_name: Text, max_confidence: Optional[float] = None
    ) -> bool:
        from rasa.core.policies.mapping_policy import MappingPolicy

        is_mapping = policy_name.endswith("_" + MappingPolicy.__name__)
        # also check if confidence is 0, than it cannot be count as prediction
        return not is_mapping or max_confidence == 0.0

    @staticmethod
    def _is_form_policy(policy_name: Text) -> bool:
        from rasa.core.policies.form_policy import FormPolicy

        return policy_name.endswith("_" + FormPolicy.__name__)

    def _pick_best_policy(
        self, predictions: Dict[Text, Prediction]
    ) -> Tuple[Optional[List[float]], Optional[Text]]:
        """Picks the best policy prediction based on probabilities and policy priority.

        Args:
            predictions: the dictionary containing policy name as keys
                         and predictions as values

        Returns:
            best_probabilities: the list of probabilities for the next actions
            best_policy_name: the name of the picked policy
        """

        best_confidence = (-1, -1)
        best_policy_name = None

        # form and mapping policies are special:
        # form should be above fallback
        # mapping should be below fallback
        # mapping is above form if it wins over fallback
        # therefore form predictions are stored separately

        form_confidence = None
        form_policy_name = None

        for policy_name, prediction in predictions.items():
            confidence = (max(prediction.probabilities), prediction.priority)
            if self._is_form_policy(policy_name):
                # store form prediction separately
                form_confidence = confidence
                form_policy_name = policy_name
            elif confidence > best_confidence:
                # pick the best policy
                best_confidence = confidence
                best_policy_name = policy_name

        if form_confidence is not None and self._is_not_mapping_policy(
            best_policy_name, best_confidence[0]
        ):
            # if mapping didn't win, check form policy predictions
            if form_confidence > best_confidence:
                best_policy_name = form_policy_name

        return predictions[best_policy_name].probabilities, best_policy_name

    def _best_policy_prediction(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[Optional[List[float]], Optional[Text]]:
        """Finds the best policy prediction.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.core.domain.Domain`

        Returns:
            probabilities: the list of probabilities for the next actions
            policy_name: the name of the picked policy
        """

        # find rejected action before running the policies
        # because some of them might add events
        rejected_action_name = None
        if len(tracker.events) > 0 and isinstance(
            tracker.events[-1], ActionExecutionRejected
        ):
            rejected_action_name = tracker.events[-1].action_name

        predictions = {
            f"policy_{i}_{type(p).__name__}": Prediction(
                p.predict_action_probabilities(tracker, domain), p.priority,
            )
            for i, p in enumerate(self.policies)
        }

        if rejected_action_name:
            logger.debug(
                f"Execution of '{rejected_action_name}' was rejected. "
                f"Setting its confidence to 0.0 in all predictions."
            )
            for prediction in predictions.values():
                prediction.probabilities[
                    domain.index_for_action(rejected_action_name)
                ] = 0.0

        return self._pick_best_policy(predictions)

    def _fallback_after_listen(
        self, domain: Domain, probabilities: List[float], policy_name: Text
    ) -> Tuple[List[float], Text]:
        """Triggers fallback if `action_listen` is predicted after a user utterance.

        This is done on the condition that:
        - a fallback policy is present,
        - there was just a user message and the predicted
          action is action_listen by a policy
          other than the MemoizationPolicy

        Args:
            domain: the :class:`rasa.core.domain.Domain`
            probabilities: the list of probabilities for the next actions
            policy_name: the name of the picked policy

        Returns:
            probabilities: the list of probabilities for the next actions
            policy_name: the name of the picked policy
        """

        fallback_idx_policy = [
            (i, p) for i, p in enumerate(self.policies) if isinstance(p, FallbackPolicy)
        ]

        if fallback_idx_policy:
            fallback_idx, fallback_policy = fallback_idx_policy[0]

            logger.debug(
                f"Action 'action_listen' was predicted after "
                f"a user message using {policy_name}. Predicting "
                f"fallback action: {fallback_policy.fallback_action_name}"
            )

            probabilities = fallback_policy.fallback_scores(domain)
            policy_name = f"policy_{fallback_idx}_{type(fallback_policy).__name__}"

        return probabilities, policy_name

    def probabilities_using_best_policy(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[Optional[List[float]], Optional[Text]]:
        """Predicts the next action the bot should take after seeing the tracker.

        Picks the best policy prediction based on probabilities and policy priority.
        Triggers fallback if `action_listen` is predicted after a user utterance.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.core.domain.Domain`

        Returns:
            best_probabilities: the list of probabilities for the next actions
            best_policy_name: the name of the picked policy
        """

        probabilities, policy_name = self._best_policy_prediction(tracker, domain)

        if (
            tracker.latest_action_name == ACTION_LISTEN_NAME
            and probabilities is not None
            and probabilities.index(max(probabilities))
            == domain.index_for_action(ACTION_LISTEN_NAME)
            and self.is_not_memo_policy(policy_name, max(probabilities))
        ):
            probabilities, policy_name = self._fallback_after_listen(
                domain, probabilities, policy_name
            )

        logger.debug(f"Predicted next action using {policy_name}")
        return probabilities, policy_name


class InvalidPolicyConfig(Exception):
    """Exception that can be raised when policy config is not valid."""

    pass
