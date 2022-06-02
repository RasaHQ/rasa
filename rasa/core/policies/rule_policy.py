from __future__ import annotations
import functools
import logging
from typing import Any, List, DefaultDict, Dict, Text, Optional, Set, Tuple, cast

from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict

from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import DOCS_URL_RULES
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.core.events import LoopInterrupted, UserUttered, ActionExecuted
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.policy import SupportedData, PolicyPrediction
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    get_active_loop_name,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.constants import (
    DEFAULT_CORE_FALLBACK_THRESHOLD,
    RULE_POLICY_PRIORITY,
    POLICY_PRIORITY,
    POLICY_MAX_HISTORY,
)
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    USER_INTENT_BACK,
    USER_INTENT_SESSION_START,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_BACK_NAME,
    RULE_SNIPPET_ACTION_NAME,
    SHOULD_NOT_BE_SET,
    PREVIOUS_ACTION,
    LOOP_NAME,
    SLOTS,
    ACTIVE_LOOP,
    RULE_ONLY_SLOTS,
    RULE_ONLY_LOOPS,
)
from rasa.shared.core.domain import InvalidDomain, State, Domain
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
import rasa.core.test
from rasa.core.training.training import create_action_fingerprints, ActionFingerprint

logger = logging.getLogger(__name__)


# These are Rasa Open Source default actions and overrule everything at any time.
DEFAULT_ACTION_MAPPINGS = {
    USER_INTENT_RESTART: ACTION_RESTART_NAME,
    USER_INTENT_BACK: ACTION_BACK_NAME,
    USER_INTENT_SESSION_START: ACTION_SESSION_START_NAME,
}

RULES = "rules"
RULES_FOR_LOOP_UNHAPPY_PATH = "rules_for_loop_unhappy_path"
RULES_NOT_IN_STORIES = "rules_not_in_stories"

LOOP_WAS_INTERRUPTED = "loop_was_interrupted"
DO_NOT_PREDICT_LOOP_ACTION = "do_not_predict_loop_action"

DEFAULT_RULES = "predicting default action with intent "
LOOP_RULES = "handling active loops and forms - "
LOOP_RULES_SEPARATOR = " - "


class InvalidRule(RasaException):
    """Exception that can be raised when rules are not valid."""

    def __init__(self, message: Text) -> None:
        super().__init__()
        self.message = message

    def __str__(self) -> Text:
        return self.message + (
            f"\nYou can find more information about the usage of "
            f"rules at {DOCS_URL_RULES}. "
        )


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITHOUT_END_TO_END_SUPPORT, is_trainable=True
)
class RulePolicy(MemoizationPolicy):
    """Policy which handles all the rules."""

    # rules use explicit json strings
    ENABLE_FEATURE_STRING_COMPRESSION = False

    # number of user inputs that is allowed in case rules are restricted
    ALLOWED_NUMBER_OF_USER_INPUTS = 1

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        Returns:
            The data type supported by this policy (ML and rule data).
        """
        return SupportedData.ML_AND_RULE_DATA

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        return {
            # Priority of the policy which is used if multiple policies predict
            # actions with the same confidence.
            POLICY_PRIORITY: RULE_POLICY_PRIORITY,
            # Confidence of the prediction if no rule matched and de-facto
            # threshold for a core fallback.
            "core_fallback_threshold": DEFAULT_CORE_FALLBACK_THRESHOLD,
            # Name of the action which should be predicted if no rule matched.
            "core_fallback_action_name": ACTION_DEFAULT_FALLBACK_NAME,
            # If `True` `core_fallback_action_name` is predicted in case no rule
            # matched.
            "enable_fallback_prediction": True,
            # If `True` rules are restricted to contain a maximum of 1
            # user message. This is used to avoid that users build a state machine
            # using the rules.
            "restrict_rules": True,
            # Whether to check for contradictions between rules and stories
            "check_for_contradictions": True,
            # the policy will use the confidence of NLU on the latest
            # user message to set the confidence of the action
            "use_nlu_confidence_as_score": False,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional[TrackerFeaturizer] = None,
        lookup: Optional[Dict] = None,
    ) -> None:
        """Initializes the policy."""
        # max history is set to `None` in order to capture any lengths of rule stories
        config[POLICY_MAX_HISTORY] = None

        super().__init__(
            config, model_storage, resource, execution_context, featurizer, lookup
        )

        self._fallback_action_name = config["core_fallback_action_name"]
        self._enable_fallback_prediction = config["enable_fallback_prediction"]
        self._check_for_contradictions = config["check_for_contradictions"]

        self._rules_sources: DefaultDict[Text, List[Tuple[Text, Text]]] = defaultdict(
            list
        )

    @classmethod
    def raise_if_incompatible_with_domain(
        cls, config: Dict[Text, Any], domain: Domain
    ) -> None:
        """Checks whether the domains action names match the configured fallback.

        Args:
            config: configuration of a `RulePolicy`
            domain: a domain
        Raises:
            `InvalidDomain` if this policy is incompatible with the domain
        """
        fallback_action_name = config.get("core_fallback_action_name", None)
        if (
            fallback_action_name
            and fallback_action_name not in domain.action_names_or_texts
        ):
            raise InvalidDomain(
                f"The fallback action '{fallback_action_name}' which was "
                f"configured for the {RulePolicy.__name__} must be "
                f"present in the domain."
            )

    @staticmethod
    def _is_rule_snippet_state(state: State) -> bool:
        prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
        return prev_action_name == RULE_SNIPPET_ACTION_NAME

    def _create_feature_key(self, states: List[State]) -> Optional[Text]:
        new_states: List[State] = []
        for state in reversed(states):
            if self._is_rule_snippet_state(state):
                # remove all states before RULE_SNIPPET_ACTION_NAME
                break
            new_states.insert(0, state)

        if not new_states:
            return None

        # we sort keys to make sure that the same states
        # represented as dictionaries have the same json strings
        return json.dumps(new_states, sort_keys=True)

    @staticmethod
    def _states_for_unhappy_loop_predictions(states: List[State]) -> List[State]:
        """Modifies the states to create feature keys for loop unhappy path conditions.

        Args:
            states: a representation of a tracker
                as a list of dictionaries containing features

        Returns:
            modified states
        """
        # leave only last 2 dialogue turns to
        # - capture previous meaningful action before action_listen
        # - ignore previous intent
        if len(states) == 1 or not states[-2].get(PREVIOUS_ACTION):
            return [states[-1]]
        else:
            return [{PREVIOUS_ACTION: states[-2][PREVIOUS_ACTION]}, states[-1]]

    @staticmethod
    def _remove_rule_snippet_predictions(lookup: Dict[Text, Text]) -> Dict[Text, Text]:
        # Delete rules if it would predict the RULE_SNIPPET_ACTION_NAME action
        return {
            feature_key: action
            for feature_key, action in lookup.items()
            if action != RULE_SNIPPET_ACTION_NAME
        }

    def _create_loop_unhappy_lookup_from_states(
        self,
        trackers_as_states: List[List[State]],
        trackers_as_actions: List[List[Text]],
    ) -> Dict[Text, Text]:
        """Creates lookup dictionary from the tracker represented as states.

        Args:
            trackers_as_states: representation of the trackers as a list of states
            trackers_as_actions: representation of the trackers as a list of actions

        Returns:
            lookup dictionary
        """
        lookup = {}
        for states, actions in zip(trackers_as_states, trackers_as_actions):
            action = actions[0]
            active_loop = get_active_loop_name(states[-1])
            # even if there are two identical feature keys
            # their loop will be the same
            if not active_loop:
                continue

            states = self._states_for_unhappy_loop_predictions(states)
            feature_key = self._create_feature_key(states)
            if not feature_key:
                continue

            # Since rule snippets and stories inside the loop contain
            # only unhappy paths, notify the loop that
            # it was predicted after an answer to a different question and
            # therefore it should not validate user input
            if (
                # loop is predicted after action_listen in unhappy path,
                # therefore no validation is needed
                is_prev_action_listen_in_state(states[-1])
                and action == active_loop
            ):
                lookup[feature_key] = LOOP_WAS_INTERRUPTED
            elif (
                # some action other than active_loop is predicted in unhappy path,
                # therefore active_loop shouldn't be predicted by the rule
                not is_prev_action_listen_in_state(states[-1])
                and action != active_loop
            ):
                lookup[feature_key] = DO_NOT_PREDICT_LOOP_ACTION
        return lookup

    def _check_rule_restriction(
        self, rule_trackers: List[TrackerWithCachedStates]
    ) -> None:
        rules_exceeding_max_user_turns = []
        for tracker in rule_trackers:
            number_of_user_uttered = sum(
                isinstance(event, UserUttered) for event in tracker.events
            )
            if number_of_user_uttered > self.ALLOWED_NUMBER_OF_USER_INPUTS:
                rules_exceeding_max_user_turns.append(tracker.sender_id)

        if rules_exceeding_max_user_turns:
            raise InvalidRule(
                f"Found rules '{', '.join(rules_exceeding_max_user_turns)}' "
                f"that contain more than {self.ALLOWED_NUMBER_OF_USER_INPUTS} "
                f"user message. Rules are not meant to hardcode a state machine. "
                f"Please use stories for these cases."
            )

    @staticmethod
    def _expected_but_missing_slots(
        fingerprint: ActionFingerprint, state: State
    ) -> Set[Text]:
        expected_slots = set(fingerprint.slots)
        current_slots = set(state.get(SLOTS, {}).keys())
        # report all slots that are expected but aren't set in current slots
        return expected_slots.difference(current_slots)

    @staticmethod
    def _check_active_loops_fingerprint(
        fingerprint: ActionFingerprint, state: State
    ) -> Set[Optional[Text]]:
        expected_active_loops = set(fingerprint.active_loop)
        # we don't use tracker.active_loop_name
        # because we need to keep should_not_be_set
        current_active_loop = state.get(ACTIVE_LOOP, {}).get(LOOP_NAME)
        if current_active_loop in expected_active_loops:
            # one of expected active loops is set
            return set()

        return expected_active_loops

    @staticmethod
    def _error_messages_from_fingerprints(
        action_name: Text,
        missing_fingerprint_slots: Set[Text],
        fingerprint_active_loops: Set[Text],
        rule_name: Text,
    ) -> List[Text]:
        error_messages = []
        if action_name and missing_fingerprint_slots:
            error_messages.append(
                f"- the action '{action_name}' in rule '{rule_name}' does not set some "
                f"of the slots that it sets in other rules. Slots not set in rule "
                f"'{rule_name}': '{', '.join(missing_fingerprint_slots)}'. Please "
                f"update the rule with an appropriate slot or if it is the last action "
                f"add 'wait_for_user_input: false' after this action."
            )
        if action_name and fingerprint_active_loops:
            # substitute `SHOULD_NOT_BE_SET` with `null` so that users
            # know what to put in their rules
            fingerprint_active_loops = set(
                "null" if active_loop == SHOULD_NOT_BE_SET else active_loop
                for active_loop in fingerprint_active_loops
            )
            # add action_name to active loop so that users
            # know what to put in their rules
            fingerprint_active_loops.add(action_name)

            error_messages.append(
                f"- the form '{action_name}' in rule '{rule_name}' does not set "
                f"the 'active_loop', that it sets in other rules: "
                f"'{', '.join(fingerprint_active_loops)}'. Please update the rule with "
                f"the appropriate 'active loop' property or if it is the last action "
                f"add 'wait_for_user_input: false' after this action."
            )
        return error_messages

    def _check_for_incomplete_rules(
        self, rule_trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> None:
        logger.debug("Started checking if some rules are incomplete.")
        # we need to use only fingerprints from rules
        rule_fingerprints = create_action_fingerprints(rule_trackers, domain)
        if not rule_fingerprints:
            return

        error_messages: List[Text] = []
        for tracker in rule_trackers:
            states = tracker.past_states(domain)
            # the last action is always action listen
            action_names = [
                state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME) for state in states[1:]
            ] + [ACTION_LISTEN_NAME]

            for state, action_name in zip(states, action_names):
                previous_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
                fingerprint = rule_fingerprints.get(previous_action_name)
                if (
                    not previous_action_name
                    or not fingerprint
                    or action_name == RULE_SNIPPET_ACTION_NAME
                    or previous_action_name == RULE_SNIPPET_ACTION_NAME
                ):
                    # do not check fingerprints for rule snippet action
                    # and don't raise if fingerprints are not satisfied
                    # for a previous action if current action is rule snippet action
                    continue

                missing_expected_slots = self._expected_but_missing_slots(
                    fingerprint, state
                )
                expected_active_loops = self._check_active_loops_fingerprint(
                    fingerprint, state
                )
                error_messages.extend(
                    self._error_messages_from_fingerprints(
                        previous_action_name,
                        missing_expected_slots,
                        expected_active_loops,
                        tracker.sender_id,
                    )
                )

        if error_messages:
            error_text = "\n".join(error_messages)
            raise InvalidRule(
                f"\nIncomplete rules found🚨\n\n{error_text}\n"
                f"Please note that if some slots or active loops should not be set "
                f"during prediction you need to explicitly set them to 'null' in the "
                f"rules."
            )

        logger.debug("Found no incompletions in rules.")

    @staticmethod
    def _get_slots_loops_from_states(
        trackers_as_states: List[List[State]],
    ) -> Tuple[Set[Text], Set[Text]]:
        slots = set()
        loops = set()
        for states in trackers_as_states:
            for state in states:
                slots.update(set(state.get(SLOTS, {}).keys()))
                # FIXME: ideally we have better annotation for State, TypedDict
                # could work but support in mypy is very limited. Dataclass are
                # another option
                active_loop = cast(Text, state.get(ACTIVE_LOOP, {}).get(LOOP_NAME))
                if active_loop:
                    loops.add(active_loop)
        return slots, loops

    def _find_rule_only_slots_loops(
        self,
        rule_trackers_as_states: List[List[State]],
        story_trackers_as_states: List[List[State]],
    ) -> Tuple[List[Text], List[Text]]:
        rule_slots, rule_loops = self._get_slots_loops_from_states(
            rule_trackers_as_states
        )
        story_slots, story_loops = self._get_slots_loops_from_states(
            story_trackers_as_states
        )

        # set is not json serializable, so convert to list
        return (
            list(rule_slots - story_slots - {SHOULD_NOT_BE_SET}),
            list(rule_loops - story_loops - {SHOULD_NOT_BE_SET}),
        )

    def _predict_next_action(
        self, tracker: TrackerWithCachedStates, domain: Domain
    ) -> Tuple[Optional[Text], Optional[Text]]:
        prediction, prediction_source = self._predict(tracker, domain)
        probabilities = prediction.probabilities
        # do not raise an error if RulePolicy didn't predict anything for stories;
        # however for rules RulePolicy should always predict an action
        predicted_action_name = None
        if (
            probabilities != self._default_predictions(domain)
            or tracker.is_rule_tracker
        ):
            predicted_action_name = domain.action_names_or_texts[
                np.argmax(probabilities)
            ]

        return predicted_action_name, prediction_source

    def _predicted_action_name(
        self, tracker: TrackerWithCachedStates, domain: Domain, gold_action_name: Text
    ) -> Tuple[Optional[Text], Optional[Text]]:
        predicted_action_name, prediction_source = self._predict_next_action(
            tracker, domain
        )
        # if there is an active_loop,
        # RulePolicy will always predict active_loop first,
        # but inside loop unhappy path there might be another action
        if (
            tracker.active_loop_name
            and predicted_action_name != gold_action_name
            and predicted_action_name == tracker.active_loop_name
        ):
            rasa.core.test.emulate_loop_rejection(tracker)
            predicted_action_name, prediction_source = self._predict_next_action(
                tracker, domain
            )

        return predicted_action_name, prediction_source

    def _collect_sources(
        self,
        tracker: TrackerWithCachedStates,
        predicted_action_name: Optional[Text],
        gold_action_name: Optional[Text],
        prediction_source: Text,
    ) -> None:
        # we need to remember which action should be predicted by the rule
        # in order to correctly output the names of the contradicting rules
        rule_name = tracker.sender_id

        if prediction_source is not None and (
            prediction_source.startswith(DEFAULT_RULES)
            or prediction_source.startswith(LOOP_RULES)
        ):
            # the real gold action contradict the one in the rules in this case
            gold_action_name = predicted_action_name
            rule_name = prediction_source

        self._rules_sources[prediction_source].append((rule_name, gold_action_name))

    @staticmethod
    def _default_sources() -> Set[Text]:
        return {
            DEFAULT_RULES + default_intent
            for default_intent in DEFAULT_ACTION_MAPPINGS.keys()
        }

    @staticmethod
    def _handling_loop_sources(domain: Domain) -> Set[Text]:
        loop_sources = set()
        for loop_name in domain.form_names:
            loop_sources.add(LOOP_RULES + loop_name)
            loop_sources.add(
                LOOP_RULES + loop_name + LOOP_RULES_SEPARATOR + ACTION_LISTEN_NAME
            )
        return loop_sources

    def _should_delete(
        self,
        prediction_source: Text,
        tracker: TrackerWithCachedStates,
        predicted_action_name: Text,
    ) -> bool:
        """Checks whether this contradiction is due to action, intent pair.

        Args:
            prediction_source: the states that result in the prediction
            tracker: the tracker that raises the contradiction

        Returns:
            true if the contradiction is a result of an action, intent pair in the rule.
        """
        if (
            # only apply to contradicting story, not rule
            tracker.is_rule_tracker
            # only apply for prediction after unpredictable action
            or prediction_source.count(PREVIOUS_ACTION) > 1
            # only apply for prediction of action_listen
            or predicted_action_name != ACTION_LISTEN_NAME
        ):
            return False
        for source in self.lookup[RULES]:
            # remove rule only if another action is predicted after action_listen
            if (
                source.startswith(prediction_source[:-2])
                and not prediction_source == source
            ):
                return True
        return False

    def _check_prediction(
        self,
        tracker: TrackerWithCachedStates,
        predicted_action_name: Optional[Text],
        gold_action_name: Text,
        prediction_source: Optional[Text],
    ) -> List[Text]:
        # FIXME: `predicted_action_name` and `prediction_source` are
        # either None together or defined together. This could be improved
        # by better typing in this class, but requires some refactoring
        if (
            not predicted_action_name
            or not prediction_source
            or predicted_action_name == gold_action_name
        ):
            return []

        if self._should_delete(prediction_source, tracker, predicted_action_name):
            self.lookup[RULES].pop(prediction_source)
            return []

        tracker_type = "rule" if tracker.is_rule_tracker else "story"
        contradicting_rules = {
            rule_name
            for rule_name, action_name in self._rules_sources[prediction_source]
            if action_name != gold_action_name
        }

        if not contradicting_rules:
            return []

        error_message = (
            f"- the prediction of the action '{gold_action_name}' in {tracker_type} "
            f"'{tracker.sender_id}' "
            f"is contradicting with rule(s) '{', '.join(contradicting_rules)}'"
        )
        # outputting predicted action 'action_default_fallback' is confusing
        if predicted_action_name != self._fallback_action_name:
            error_message += f" which predicted action '{predicted_action_name}'"

        return [error_message + "."]

    def _run_prediction_on_trackers(
        self,
        trackers: List[TrackerWithCachedStates],
        domain: Domain,
        collect_sources: bool,
    ) -> Tuple[List[Text], Set[Optional[Text]]]:
        if collect_sources:
            self._rules_sources = defaultdict(list)

        error_messages = []
        rules_used_in_stories = set()
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:
            running_tracker = tracker.init_copy()
            running_tracker.sender_id = tracker.sender_id
            # the first action is always unpredictable
            next_action_is_unpredictable = True
            for event in tracker.applied_events():
                if not isinstance(event, ActionExecuted):
                    running_tracker.update(event)
                    continue

                if event.action_name == RULE_SNIPPET_ACTION_NAME:
                    # notify that the action after RULE_SNIPPET_ACTION_NAME is
                    # unpredictable
                    next_action_is_unpredictable = True
                    running_tracker.update(event)
                    continue

                # do not run prediction on unpredictable actions
                if next_action_is_unpredictable or event.unpredictable:
                    next_action_is_unpredictable = False  # reset unpredictability
                    running_tracker.update(event)
                    continue

                gold_action_name = event.action_name or event.action_text
                predicted_action_name, prediction_source = self._predicted_action_name(
                    running_tracker, domain, gold_action_name
                )
                if collect_sources:
                    if prediction_source:
                        self._collect_sources(
                            running_tracker,
                            predicted_action_name,
                            gold_action_name,
                            prediction_source,
                        )
                else:
                    # to be able to remove only rules turns from the dialogue history
                    # for ML policies,
                    # we need to know which rules were used in ML trackers
                    if (
                        not tracker.is_rule_tracker
                        and predicted_action_name == gold_action_name
                    ):
                        rules_used_in_stories.add(prediction_source)

                    error_messages += self._check_prediction(
                        running_tracker,
                        predicted_action_name,
                        gold_action_name,
                        prediction_source,
                    )

                running_tracker.update(event)

        return error_messages, rules_used_in_stories

    def _collect_rule_sources(
        self, rule_trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> None:
        self._run_prediction_on_trackers(rule_trackers, domain, collect_sources=True)

    def _find_contradicting_and_used_in_stories_rules(
        self, trackers: List[TrackerWithCachedStates], domain: Domain
    ) -> Tuple[List[Text], Set[Optional[Text]]]:
        return self._run_prediction_on_trackers(trackers, domain, collect_sources=False)

    def _analyze_rules(
        self,
        rule_trackers: List[TrackerWithCachedStates],
        all_trackers: List[TrackerWithCachedStates],
        domain: Domain,
    ) -> List[Text]:
        """Analyzes learned rules by running prediction on training trackers.

        This method collects error messages for contradicting rules
        and creates the lookup for rules that are not present in the stories.

        Args:
            rule_trackers: The list of the rule trackers.
            all_trackers: The list of all trackers.
            domain: The domain.

        Returns:
             Rules that are not present in the stories.
        """
        logger.debug("Started checking rules and stories for contradictions.")
        # during training we run `predict_action_probabilities` to check for
        # contradicting rules.
        # We silent prediction debug to avoid too many logs during these checks.
        logger_level = logger.level
        logger.setLevel(logging.WARNING)

        # we need to run prediction on rule trackers twice, because we need to collect
        # the information about which rule snippets contributed to the learned rules
        self._collect_rule_sources(rule_trackers, domain)
        (
            error_messages,
            rules_used_in_stories,
        ) = self._find_contradicting_and_used_in_stories_rules(all_trackers, domain)

        logger.setLevel(logger_level)  # reset logger level
        if error_messages:
            error_text = "\n".join(error_messages)
            raise InvalidRule(
                f"\nContradicting rules or stories found 🚨\n\n{error_text}\n"
                f"Please update your stories and rules so that they don't contradict "
                f"each other."
            )

        logger.debug("Found no contradicting rules.")
        all_rules = (
            set(self._rules_sources.keys())
            | self._default_sources()
            | self._handling_loop_sources(domain)
        )
        # set is not json serializable, so convert to list
        return list(all_rules - rules_used_in_stories)

    def _create_lookup_from_trackers(
        self,
        rule_trackers: List[TrackerWithCachedStates],
        story_trackers: List[TrackerWithCachedStates],
        domain: Domain,
    ) -> None:
        (
            rule_trackers_as_states,
            rule_trackers_as_actions,
        ) = self.featurizer.training_states_and_labels(
            rule_trackers, domain, omit_unset_slots=True
        )

        rules_lookup = self._create_lookup_from_states(
            rule_trackers_as_states, rule_trackers_as_actions
        )
        self.lookup[RULES] = self._remove_rule_snippet_predictions(rules_lookup)

        (
            story_trackers_as_states,
            story_trackers_as_actions,
        ) = self.featurizer.training_states_and_labels(story_trackers, domain)

        if self._check_for_contradictions:
            (
                self.lookup[RULE_ONLY_SLOTS],
                self.lookup[RULE_ONLY_LOOPS],
            ) = self._find_rule_only_slots_loops(
                rule_trackers_as_states, story_trackers_as_states
            )

        # use all trackers to find negative rules in unhappy paths
        trackers_as_states = rule_trackers_as_states + story_trackers_as_states
        trackers_as_actions = rule_trackers_as_actions + story_trackers_as_actions

        # negative rules are not anti-rules, they are auxiliary to actual rules
        self.lookup[
            RULES_FOR_LOOP_UNHAPPY_PATH
        ] = self._create_loop_unhappy_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        **kwargs: Any,
    ) -> Resource:
        """Trains the policy on given training trackers.

        Args:
            training_trackers: The list of the trackers.
            domain: The domain.

        Returns:
            The resource which can be used to load the trained policy.
        """
        self.raise_if_incompatible_with_domain(self.config, domain)

        # only consider original trackers (no augmented ones)
        training_trackers = [
            t for t in training_trackers if not getattr(t, "is_augmented", False)
        ]
        # trackers from rule-based training data
        rule_trackers = [t for t in training_trackers if t.is_rule_tracker]
        if self.config["restrict_rules"]:
            self._check_rule_restriction(rule_trackers)
        if self._check_for_contradictions:
            self._check_for_incomplete_rules(rule_trackers, domain)

        # trackers from ML-based training data
        story_trackers = [t for t in training_trackers if not t.is_rule_tracker]

        self._create_lookup_from_trackers(rule_trackers, story_trackers, domain)

        # make this configurable because checking might take a lot of time
        if self._check_for_contradictions:
            # using trackers here might not be the most efficient way, however
            # it allows us to directly test `predict_action_probabilities` method
            self.lookup[RULES_NOT_IN_STORIES] = self._analyze_rules(
                rule_trackers, training_trackers, domain
            )

        logger.debug(f"Memorized '{len(self.lookup[RULES])}' unique rules.")

        self.persist()

        return self._resource

    @staticmethod
    def _does_rule_match_state(rule_state: State, conversation_state: State) -> bool:
        for state_type, rule_sub_state in rule_state.items():
            conversation_sub_state = conversation_state.get(state_type, {})
            for key, value_from_rules in rule_sub_state.items():
                if isinstance(value_from_rules, list):
                    # json dumps and loads tuples as lists,
                    # so we need to convert them back
                    value_from_rules = tuple(value_from_rules)
                value_from_conversation = conversation_sub_state.get(key)
                if (
                    # value should be set, therefore
                    # check whether it is the same as in the state
                    value_from_rules
                    and value_from_rules != SHOULD_NOT_BE_SET
                    and value_from_conversation != value_from_rules
                ) or (
                    # value shouldn't be set, therefore
                    # it should be None or non existent in the state
                    value_from_rules == SHOULD_NOT_BE_SET
                    and value_from_conversation
                    # during training `SHOULD_NOT_BE_SET` is provided. Hence, we also
                    # have to check for the value of the slot state
                    and value_from_conversation != SHOULD_NOT_BE_SET
                ):
                    return False

        return True

    @staticmethod
    # This function is called a lot (e.g. for checking contradictions) so we cache
    # its results.
    @functools.lru_cache(maxsize=1000)
    def _rule_key_to_state(rule_key: Text) -> List[State]:
        return json.loads(rule_key)

    def _is_rule_applicable(
        self, rule_key: Text, turn_index: int, conversation_state: State
    ) -> bool:
        """Checks if rule is satisfied with current state at turn.

        Args:
            rule_key: the textual representation of learned rule
            turn_index: index of a current dialogue turn
            conversation_state: the state that corresponds to turn_index

        Returns:
            a boolean that says whether the rule is applicable to current state
        """
        # turn_index goes back in time
        reversed_rule_states = list(reversed(self._rule_key_to_state(rule_key)))

        # the rule must be applicable because we got (without any applicability issues)
        # further in the conversation history than the rule's length
        if turn_index >= len(reversed_rule_states):
            return True

        # a state has previous action if and only if it is not a conversation start
        # state
        current_previous_action = conversation_state.get(PREVIOUS_ACTION)
        rule_previous_action = reversed_rule_states[turn_index].get(PREVIOUS_ACTION)

        # current conversation state and rule state are conversation starters.
        # any slots with initial_value set will necessarily be in both states and don't
        # need to be checked.
        if not rule_previous_action and not current_previous_action:
            return True

        # current rule state is a conversation starter (due to conversation_start: true)
        # but current conversation state is not.
        # or
        # current conversation state is a starter
        # but current rule state is not.
        if not rule_previous_action or not current_previous_action:
            return False

        # check: current rule state features are present in current conversation state
        return self._does_rule_match_state(
            reversed_rule_states[turn_index], conversation_state
        )

    def _get_possible_keys(
        self, lookup: Dict[Text, Text], states: List[State]
    ) -> Set[Text]:
        possible_keys = set(lookup.keys())
        for i, state in enumerate(reversed(states)):
            # find rule keys that correspond to current state
            possible_keys = set(
                filter(
                    lambda _key: self._is_rule_applicable(_key, i, state), possible_keys
                )
            )
        return possible_keys

    @staticmethod
    def _find_action_from_default_actions(
        tracker: DialogueStateTracker,
    ) -> Tuple[Optional[Text], Optional[Text]]:
        if (
            not tracker.latest_action_name == ACTION_LISTEN_NAME
            or not tracker.latest_message
        ):
            return None, None

        intent_name = tracker.latest_message.intent.get(INTENT_NAME_KEY)
        if intent_name is None:
            return None, None

        default_action_name = DEFAULT_ACTION_MAPPINGS.get(intent_name)
        if default_action_name is None:
            return None, None

        logger.debug(f"Predicted default action '{default_action_name}'.")
        return (
            default_action_name,
            # create prediction source that corresponds to one of
            # default prediction sources in `_default_sources()`
            DEFAULT_RULES + intent_name,
        )

    @staticmethod
    def _find_action_from_loop_happy_path(
        tracker: DialogueStateTracker,
    ) -> Tuple[Optional[Text], Optional[Text]]:

        active_loop_name = tracker.active_loop_name
        if active_loop_name is None:
            return None, None

        active_loop_rejected = tracker.is_active_loop_rejected
        should_predict_loop = (
            not active_loop_rejected
            and tracker.latest_action
            and tracker.latest_action.get(ACTION_NAME) != active_loop_name
        )
        should_predict_listen = (
            not active_loop_rejected and tracker.latest_action_name == active_loop_name
        )

        if should_predict_loop:
            logger.debug(f"Predicted loop '{active_loop_name}'.")
            return active_loop_name, LOOP_RULES + active_loop_name

        # predict `action_listen` if loop action was run successfully
        if should_predict_listen:
            logger.debug(
                f"Predicted '{ACTION_LISTEN_NAME}' after loop '{active_loop_name}'."
            )
            return (
                ACTION_LISTEN_NAME,
                (
                    f"{LOOP_RULES}{active_loop_name}"
                    f"{LOOP_RULES_SEPARATOR}{ACTION_LISTEN_NAME}"
                ),
            )

        return None, None

    def _find_action_from_rules(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool,
    ) -> Tuple[Optional[Text], Optional[Text], bool]:
        """Predicts the next action based on the memoized rules.

        Args:
            tracker: The current conversation tracker.
            domain: The domain of the current model.
            use_text_for_last_user_input: `True` if text of last user message
                should be used for the prediction. `False` if intent should be used.

        Returns:
            A tuple of the predicted action name or text (or `None` if no matching rule
            was found), a description of the matching rule, and `True` if a loop action
            was predicted after the loop has been in an unhappy path before.
        """
        if (
            use_text_for_last_user_input
            and not tracker.latest_action_name == ACTION_LISTEN_NAME
        ):
            # make text prediction only directly after user utterance
            # because we've otherwise already decided whether to use
            # the text or the intent
            return None, None, False

        states = self._prediction_states(
            tracker,
            domain,
            use_text_for_last_user_input,
            rule_only_data=self._get_rule_only_data(),
        )

        current_states = self.format_tracker_states(states)
        logger.debug(f"Current tracker state:{current_states}")

        # Tracks if we are returning after an unhappy loop path. If this becomes `True`
        # the policy returns an event which notifies the loop action that it
        # is returning after an unhappy path. For example, the `FormAction` uses this
        # to skip the validation of slots for its first execution after an unhappy path.
        returning_from_unhappy_path = False

        rule_keys = self._get_possible_keys(self.lookup[RULES], states)
        predicted_action_name = None
        best_rule_key = ""
        if rule_keys:
            # if there are several rules,
            # it should mean that some rule is a subset of another rule
            # therefore we pick a rule of maximum length
            best_rule_key = max(rule_keys, key=len)
            predicted_action_name = self.lookup[RULES].get(best_rule_key)

        active_loop_name = tracker.active_loop_name
        if active_loop_name:
            # find rules for unhappy path of the loop
            loop_unhappy_keys = self._get_possible_keys(
                self.lookup[RULES_FOR_LOOP_UNHAPPY_PATH], states
            )
            # there could be several unhappy path conditions
            unhappy_path_conditions = [
                self.lookup[RULES_FOR_LOOP_UNHAPPY_PATH].get(key)
                for key in loop_unhappy_keys
            ]

            # Check if a rule that predicted action_listen
            # was applied inside the loop.
            # Rules might not explicitly switch back to the loop.
            # Hence, we have to take care of that.
            predicted_listen_from_general_rule = (
                predicted_action_name == ACTION_LISTEN_NAME
                and not get_active_loop_name(self._rule_key_to_state(best_rule_key)[-1])
            )
            if predicted_listen_from_general_rule:
                if DO_NOT_PREDICT_LOOP_ACTION not in unhappy_path_conditions:
                    # negative rules don't contain a key that corresponds to
                    # the fact that active_loop shouldn't be predicted
                    logger.debug(
                        f"Predicted loop '{active_loop_name}' by overwriting "
                        f"'{ACTION_LISTEN_NAME}' predicted by general rule."
                    )
                    return (
                        active_loop_name,
                        best_rule_key,
                        returning_from_unhappy_path,
                    )

                # do not predict anything
                predicted_action_name = None

            if LOOP_WAS_INTERRUPTED in unhappy_path_conditions:
                logger.debug(
                    "Returning from unhappy path. Loop will be notified that "
                    "it was interrupted."
                )
                returning_from_unhappy_path = True

        if predicted_action_name is not None:
            logger.debug(
                f"There is a rule for the next action '{predicted_action_name}'."
            )
        else:
            logger.debug("There is no applicable rule.")

        # if we didn't predict anything from the rules, then the feature key created
        # from states can be used as an indicator that this state will lead to fallback
        return (
            predicted_action_name,
            best_rule_key or self._create_feature_key(states),
            returning_from_unhappy_path,
        )

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action (see parent class for more information)."""
        prediction, _ = self._predict(tracker, domain)
        return prediction

    def _predict(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[PolicyPrediction, Optional[Text]]:
        (
            rules_action_name_from_text,
            prediction_source_from_text,
            returning_from_unhappy_path_from_text,
        ) = self._find_action_from_rules(
            tracker, domain, use_text_for_last_user_input=True
        )

        # Rasa Open Source default actions overrule anything. If users want to achieve
        # the same, they need to write a rule or make sure that their loop rejects
        # accordingly.
        (
            default_action_name,
            default_prediction_source,
        ) = self._find_action_from_default_actions(tracker)

        # text has priority over intents including default,
        # however loop happy path has priority over rules prediction
        if default_action_name and not rules_action_name_from_text:
            return (
                self._rule_prediction(
                    self._prediction_result(default_action_name, tracker, domain),
                    default_prediction_source,
                ),
                default_prediction_source,
            )

        # A loop has priority over any other rule except defaults.
        # The rules or any other prediction will be applied only if a loop was rejected.
        # If we are in a loop, and the loop didn't run previously or rejected, we can
        # simply force predict the loop.
        (
            loop_happy_path_action_name,
            loop_happy_path_prediction_source,
        ) = self._find_action_from_loop_happy_path(tracker)
        if loop_happy_path_action_name:
            # this prediction doesn't use user input
            # and happy user input anyhow should be ignored during featurization
            return (
                self._rule_prediction(
                    self._prediction_result(
                        loop_happy_path_action_name, tracker, domain
                    ),
                    loop_happy_path_prediction_source,
                    is_no_user_prediction=True,
                ),
                loop_happy_path_prediction_source,
            )

        # predict rules from text first
        if rules_action_name_from_text:
            return (
                self._rule_prediction(
                    self._prediction_result(
                        rules_action_name_from_text, tracker, domain
                    ),
                    prediction_source_from_text,
                    returning_from_unhappy_path=returning_from_unhappy_path_from_text,
                    is_end_to_end_prediction=True,
                ),
                prediction_source_from_text,
            )

        (
            rules_action_name_from_intent,
            # we want to remember the source even if rules didn't predict any action
            prediction_source_from_intent,
            returning_from_unhappy_path_from_intent,
        ) = self._find_action_from_rules(
            tracker, domain, use_text_for_last_user_input=False
        )
        if rules_action_name_from_intent:
            probabilities = self._prediction_result(
                rules_action_name_from_intent, tracker, domain
            )
        else:
            probabilities = self._default_predictions(domain)

        return (
            self._rule_prediction(
                probabilities,
                prediction_source_from_intent,
                returning_from_unhappy_path=(
                    # returning_from_unhappy_path is a negative condition,
                    # so `or` should be applied
                    returning_from_unhappy_path_from_text
                    or returning_from_unhappy_path_from_intent
                ),
                is_end_to_end_prediction=False,
            ),
            prediction_source_from_intent,
        )

    def _rule_prediction(
        self,
        probabilities: List[float],
        prediction_source: Text,
        returning_from_unhappy_path: bool = False,
        is_end_to_end_prediction: bool = False,
        is_no_user_prediction: bool = False,
    ) -> PolicyPrediction:
        return PolicyPrediction(
            probabilities,
            self.__class__.__name__,
            self.priority,
            events=[LoopInterrupted(True)] if returning_from_unhappy_path else [],
            is_end_to_end_prediction=is_end_to_end_prediction,
            is_no_user_prediction=is_no_user_prediction,
            hide_rule_turn=(
                True
                if prediction_source in self.lookup.get(RULES_NOT_IN_STORIES, [])
                else False
            ),
        )

    def _default_predictions(self, domain: Domain) -> List[float]:
        result = super()._default_predictions(domain)

        if self._enable_fallback_prediction:
            result[domain.index_for_action(self._fallback_action_name)] = self.config[
                "core_fallback_threshold"
            ]
        return result

    def persist(self) -> None:
        """Persists trained `RulePolicy`."""
        super().persist()
        with self._model_storage.write_to(self._resource) as directory:
            rule_only_data = self._get_rule_only_data()
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                directory / "rule_only_data.json", rule_only_data
            )

    def _metadata(self) -> Dict[Text, Any]:
        return {"lookup": self.lookup}

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "rule_policy.json"

    def _get_rule_only_data(self) -> Dict[Text, Any]:
        """Gets the slots and loops that are used only in rule data.

        Returns:
            Slots and loops that are used only in rule data.
        """
        return {
            key: self.lookup.get(key, []) for key in [RULE_ONLY_SLOTS, RULE_ONLY_LOOPS]
        }
