import importlib
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Text, Optional, Any, List, Dict, Tuple, NamedTuple, Union

import rasa.core
import rasa.core.training.training
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.utils.io
from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.shared.constants import (
    DOCS_URL_RULES,
    DOCS_URL_POLICIES,
    DOCS_URL_MIGRATION_GUIDE,
    DEFAULT_CONFIG_PATH,
)
from rasa.shared.core.constants import (
    USER_INTENT_BACK,
    USER_INTENT_RESTART,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
)
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.core.events import ActionExecutionRejected
from rasa.core.exceptions import UnsupportedDialogueModelError
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.policies.policy import Policy, SupportedData
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core import registry

logger = logging.getLogger(__name__)


class PolicyEnsemble:
    versioned_packages = ["rasa", "tensorflow", "sklearn"]

    def __init__(
        self,
        policies: List[Policy],
        action_fingerprints: Optional[Dict[Any, Dict[Text, List]]] = None,
    ) -> None:
        self.policies = policies
        self.date_trained = None

        self.action_fingerprints = action_fingerprints

        self._check_priorities()
        self._check_for_important_policies()

    def _check_for_important_policies(self) -> None:
        from rasa.core.policies.mapping_policy import MappingPolicy

        if not any(
            isinstance(policy, (MappingPolicy, RulePolicy)) for policy in self.policies
        ):
            logger.info(
                f"MappingPolicy not included in policy ensemble. Default intents "
                f"'{USER_INTENT_RESTART} and {USER_INTENT_BACK} will not trigger "
                f"actions '{ACTION_RESTART_NAME}' and '{ACTION_BACK_NAME}'."
            )

    @staticmethod
    def check_domain_ensemble_compatibility(
        ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        """Check for elements that only work with certain policy/domain combinations."""

        from rasa.core.policies.mapping_policy import MappingPolicy
        from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy

        policies_needing_validation = [
            MappingPolicy,
            TwoStageFallbackPolicy,
            RulePolicy,
        ]
        for policy in policies_needing_validation:
            policy.validate_against_domain(ensemble, domain)

        _check_policy_for_forms_available(domain, ensemble)

    def _check_priorities(self) -> None:
        """Checks for duplicate policy priorities within PolicyEnsemble."""

        priority_dict = defaultdict(list)
        for p in self.policies:
            priority_dict[p.priority].append(type(p).__name__)

        for k, v in priority_dict.items():
            if len(v) > 1:
                rasa.shared.utils.io.raise_warning(
                    f"Found policies {v} with same priority {k} "
                    f"in PolicyEnsemble. When personalizing "
                    f"priorities, be sure to give all policies "
                    f"different priorities.",
                    docs=DOCS_URL_POLICIES,
                )

    def _policy_ensemble_contains_policy_with_rules_support(self) -> bool:
        """Determine whether the policy ensemble contains at least one policy
        supporting rule-based data.

        Returns:
            Whether or not the policy ensemble contains at least one policy that
            supports rule-based data.
        """
        return any(
            policy.supported_data()
            in [SupportedData.RULE_DATA, SupportedData.ML_AND_RULE_DATA]
            for policy in self.policies
        )

    @staticmethod
    def _training_trackers_contain_rule_trackers(
        training_trackers: List[DialogueStateTracker],
    ) -> bool:
        """Determine whether there are rule-based training trackers.

        Args:
            training_trackers: Trackers to inspect.

        Returns:
            Whether or not any of the supplied training trackers contain rule-based
            data.
        """
        return any(tracker.is_rule_tracker for tracker in training_trackers)

    def _emit_rule_policy_warning(
        self, training_trackers: List[DialogueStateTracker]
    ) -> None:
        """Emit `UserWarning`s about missing rule-based data."""
        is_rules_consuming_policy_available = (
            self._policy_ensemble_contains_policy_with_rules_support()
        )
        training_trackers_contain_rule_trackers = self._training_trackers_contain_rule_trackers(
            training_trackers
        )

        if (
            is_rules_consuming_policy_available
            and not training_trackers_contain_rule_trackers
        ):
            rasa.shared.utils.io.raise_warning(
                f"Found a rule-based policy in your pipeline but "
                f"no rule-based training data. Please add rule-based "
                f"stories to your training data or "
                f"remove the rule-based policy (`{RulePolicy.__name__}`) from your "
                f"your pipeline.",
                docs=DOCS_URL_RULES,
            )
        elif (
            not is_rules_consuming_policy_available
            and training_trackers_contain_rule_trackers
        ):
            rasa.shared.utils.io.raise_warning(
                f"Found rule-based training data but no policy supporting rule-based "
                f"data. Please add `{RulePolicy.__name__}` or another rule-supporting "
                f"policy to the `policies` section in `{DEFAULT_CONFIG_PATH}`.",
                docs=DOCS_URL_RULES,
            )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        if training_trackers:
            self._emit_rule_policy_warning(training_trackers)

            for policy in self.policies:
                trackers_to_train = SupportedData.trackers_for_policy(
                    policy, training_trackers
                )
                policy.train(
                    trackers_to_train, domain, interpreter=interpreter, **kwargs
                )

            self.action_fingerprints = rasa.core.training.training.create_action_fingerprints(
                training_trackers, domain
            )
        else:
            logger.info("Skipped training, because there are no training samples.")

        self.date_trained = datetime.now().strftime("%Y%m%d-%H%M%S")

    def probabilities_using_best_policy(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> Tuple[List[float], Optional[Text]]:
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
        rasa.shared.utils.io.create_directory_for_file(domain_spec_path)

        policy_names = [
            rasa.shared.utils.common.module_path_from_instance(p) for p in self.policies
        ]

        metadata = {
            "action_fingerprints": self.action_fingerprints,
            "python": ".".join([str(s) for s in sys.version_info[:3]]),
            "max_histories": self._max_histories(),
            "ensemble_name": self.__module__ + "." + self.__class__.__name__,
            "policy_names": policy_names,
            "trained_at": self.date_trained,
        }

        self._add_package_version_info(metadata)

        rasa.shared.utils.io.dump_obj_as_json_to_file(domain_spec_path, metadata)

    def persist(self, path: Union[Text, Path]) -> None:
        """Persists the policy to storage."""

        self._persist_metadata(path)

        for i, policy in enumerate(self.policies):
            dir_name = "policy_{}_{}".format(i, type(policy).__name__)
            policy_path = Path(path) / dir_name
            policy.persist(policy_path)

    @classmethod
    def load_metadata(cls, path) -> Any:
        metadata_path = os.path.join(path, "metadata.json")
        metadata = json.loads(
            rasa.shared.utils.io.read_file(os.path.abspath(metadata_path))
        )
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
    def load(cls, path: Union[Text, Path]) -> "PolicyEnsemble":
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
        ensemble_cls = rasa.shared.utils.common.class_from_module_path(
            metadata["ensemble_name"]
        )
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

            policy_name = policy.pop("name")
            try:
                constr_func = registry.policy_from_module_path(policy_name)
                try:
                    policy_object = constr_func(**policy)
                except TypeError as e:
                    raise Exception(f"Could not initialize {policy_name}. {e}")
                parsed_policies.append(policy_object)
            except (ImportError, AttributeError):
                raise InvalidPolicyConfig(
                    f"Module for policy '{policy_name}' could not "
                    f"be loaded. Please make sure the "
                    f"name is a valid policy."
                )

        cls._check_if_rule_policy_used_with_rule_like_policies(parsed_policies)

        return parsed_policies

    @classmethod
    def get_featurizer_from_dict(cls, policy) -> Tuple[Any, Any]:
        # policy can have only 1 featurizer
        if len(policy["featurizer"]) > 1:
            raise InvalidPolicyConfig(
                f"Every policy can only have 1 featurizer "
                f"but '{policy.get('name')}' "
                f"uses {len(policy['featurizer'])} featurizers."
            )
        featurizer_config = policy["featurizer"][0]
        featurizer_name = featurizer_config.pop("name")
        featurizer_func = registry.featurizer_from_module_path(featurizer_name)

        return featurizer_func, featurizer_config

    @classmethod
    def get_state_featurizer_from_dict(cls, featurizer_config) -> Tuple[Any, Any]:
        # featurizer can have only 1 state featurizer
        if len(featurizer_config["state_featurizer"]) > 1:
            raise InvalidPolicyConfig(
                f"Every featurizer can only have 1 state "
                f"featurizer but one of the featurizers uses "
                f"{len(featurizer_config['state_featurizer'])}."
            )
        state_featurizer_config = featurizer_config["state_featurizer"][0]
        state_featurizer_name = state_featurizer_config.pop("name")
        state_featurizer_func = registry.state_featurizer_from_module_path(
            state_featurizer_name
        )

        return state_featurizer_func, state_featurizer_config

    @staticmethod
    def _check_if_rule_policy_used_with_rule_like_policies(
        policies: List[Policy],
    ) -> None:
        if not any(isinstance(policy, RulePolicy) for policy in policies):
            return

        from rasa.core.policies.mapping_policy import MappingPolicy
        from rasa.core.policies.form_policy import FormPolicy
        from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy

        policies_not_be_used_with_rule_policy = (
            MappingPolicy,
            FormPolicy,
            FallbackPolicy,
            TwoStageFallbackPolicy,
        )

        if any(
            isinstance(policy, policies_not_be_used_with_rule_policy)
            for policy in policies
        ):
            rasa.shared.utils.io.raise_warning(
                f"It is not recommended to use the '{RulePolicy.__name__}' with "
                f"other policies which implement rule-like "
                f"behavior. It is highly recommended to migrate all deprecated "
                f"policies to use the '{RulePolicy.__name__}'. Note that the "
                f"'{RulePolicy.__name__}' will supersede the predictions of the "
                f"deprecated policies if the confidence levels of the predictions are "
                f"equal.",
                docs=DOCS_URL_MIGRATION_GUIDE,
            )


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
    ) -> Tuple[List[float], Optional[Text]]:
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
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Tuple[List[float], Optional[Text]]:
        """Finds the best policy prediction.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: Interpreter which may be used by the policies to create
                additional features.

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
            f"policy_{i}_{type(p).__name__}": self._get_prediction(
                p, tracker, domain, interpreter
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

    @staticmethod
    def _get_prediction(
        policy: Policy,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Prediction:
        number_of_arguments_in_rasa_1_0 = 2
        arguments = rasa.shared.utils.common.arguments_of(
            policy.predict_action_probabilities
        )
        if (
            len(arguments) > number_of_arguments_in_rasa_1_0
            and "interpreter" in arguments
        ):
            probabilities = policy.predict_action_probabilities(
                tracker, domain, interpreter
            )
        else:
            rasa.shared.utils.io.raise_warning(
                "The function `predict_action_probabilities` of "
                "the `Policy` interface was changed to support "
                "additional parameters. Please make sure to "
                "adapt your custom `Policy` implementation.",
                category=DeprecationWarning,
            )
            probabilities = policy.predict_action_probabilities(
                tracker, domain, RegexInterpreter()
            )

        return Prediction(probabilities, policy.priority)

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
            domain: the :class:`rasa.shared.core.domain.Domain`
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
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> Tuple[List[float], Optional[Text]]:
        """Predicts the next action the bot should take after seeing the tracker.

        Picks the best policy prediction based on probabilities and policy priority.
        Triggers fallback if `action_listen` is predicted after a user utterance.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: Interpreter which may be used by the policies to create
                additional features.

        Returns:
            best_probabilities: the list of probabilities for the next actions
            best_policy_name: the name of the picked policy
        """

        probabilities, policy_name = self._best_policy_prediction(
            tracker, domain, interpreter
        )

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


def _check_policy_for_forms_available(
    domain: Domain, ensemble: Optional["PolicyEnsemble"]
) -> None:
    if not ensemble:
        return

    from rasa.core.policies.form_policy import FormPolicy

    suited_policies_for_forms = (FormPolicy, RulePolicy)

    has_policy_for_forms = ensemble is not None and any(
        isinstance(policy, suited_policies_for_forms) for policy in ensemble.policies
    )

    if domain.form_names and not has_policy_for_forms:
        raise InvalidDomain(
            "You have defined a form action, but haven't added the "
            "FormPolicy to your policy ensemble. Either remove all "
            "forms from your domain or exclude the FormPolicy from your "
            "policy configuration."
        )


class InvalidPolicyConfig(RasaException):
    """Exception that can be raised when policy config is not valid."""

    pass
