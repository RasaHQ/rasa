import logging
from typing import Any, List, Dict, Text, Optional, Set, Tuple, TYPE_CHECKING

from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict

from rasa.shared.constants import DOCS_URL_RULES
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.core.events import (
    LoopInterrupted,
    UserUttered,
    ActionExecuted,
)
from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
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
    LOOP_REJECTED,
    LOOP_NAME,
    SLOTS,
    ACTIVE_LOOP,
    RULE_ONLY_SLOTS,
    RULE_ONLY_LOOPS,
)
from rasa.shared.core.domain import InvalidDomain, State, Domain
from rasa.shared.nlu.constants import ACTION_NAME, INTENT_NAME_KEY
import rasa.core.test
import rasa.core.training.training


if TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble

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


class StateMachinePolicy(MemoizationPolicy):
    """Policy which handles all the rules"""

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

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = RULE_POLICY_PRIORITY,
        lookup: Optional[Dict] = None,
        core_fallback_threshold: float = DEFAULT_CORE_FALLBACK_THRESHOLD,
        core_fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        enable_fallback_prediction: bool = True,
        restrict_rules: bool = True,
        check_for_contradictions: bool = True,
        **kwargs: Any,
    ) -> None:
        """Create a `StateMachinePolicy` object.

        Args:
            featurizer: `Featurizer` which is used to convert conversation states to
                features.
            priority: Priority of the policy which is used if multiple policies predict
                actions with the same confidence.
            lookup: Lookup table which is used to pick matching rules for a conversation
                state.
            core_fallback_threshold: Confidence of the prediction if no rule matched
                and de-facto threshold for a core fallback.
            core_fallback_action_name: Name of the action which should be predicted
                if no rule matched.
            enable_fallback_prediction: If `True` `core_fallback_action_name` is
                predicted in case no rule matched.
            restrict_rules: If `True` rules are restricted to contain a maximum of 1
                user message. This is used to avoid that users build a state machine
                using the rules.
            check_for_contradictions: Check for contradictions.
        """
        self._core_fallback_threshold = core_fallback_threshold
        self._fallback_action_name = core_fallback_action_name
        self._enable_fallback_prediction = enable_fallback_prediction
        self._restrict_rules = restrict_rules
        self._check_for_contradictions = check_for_contradictions

        self._rules_sources = None

        # max history is set to `None` in order to capture any lengths of rule stories
        super().__init__(
            featurizer=featurizer,
            priority=priority,
            max_history=None,
            lookup=lookup,
            **kwargs,
        )

    @classmethod
    def validate_against_domain(
        cls, ensemble: Optional["PolicyEnsemble"], domain: Optional[Domain]
    ) -> None:
        if ensemble is None:
            return

        rule_policy = next(
            (p for p in ensemble.policies if isinstance(p, StateMachinePolicy)),
            None,
        )
        if not rule_policy or not rule_policy._enable_fallback_prediction:
            return

        if (
            domain is None
            or rule_policy._fallback_action_name not in domain.action_names_or_texts
        ):
            raise InvalidDomain(
                f"The fallback action '{rule_policy._fallback_action_name}' which was "
                f"configured for the {StateMachinePolicy.__name__} must be present in the "
                f"domain."
            )

    @staticmethod
    def _is_rule_snippet_state(state: State) -> bool:
        prev_action_name = state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)
        return prev_action_name == RULE_SNIPPET_ACTION_NAME

    def _create_feature_key(self, states: List[State]) -> Optional[Text]:
        new_states = []
        for state in reversed(states):
            if self._is_rule_snippet_state(state):
                # remove all states before RULE_SNIPPET_ACTION_NAME
                break
            new_states.insert(0, state)

        if not new_states:
            return

        # we sort keys to make sure that the same states
        # represented as dictionaries have the same json strings
        return json.dumps(new_states, sort_keys=True)

    @staticmethod
    def _states_for_unhappy_loop_predictions(
        states: List[State],
    ) -> List[State]:
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
        fingerprint: Dict[Text, List[Text]], state: State
    ) -> Set[Text]:
        expected_slots = set(fingerprint.get(SLOTS, {}))
        current_slots = set(state.get(SLOTS, {}).keys())
        # report all slots that are expected but aren't set in current slots
        return expected_slots.difference(current_slots)

    @staticmethod
    def _check_active_loops_fingerprint(
        fingerprint: Dict[Text, List[Text]], state: State
    ) -> Set[Text]:
        expected_active_loops = set(fingerprint.get(ACTIVE_LOOP, {}))
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
        rule_fingerprints = rasa.core.training.training.create_action_fingerprints(
            rule_trackers, domain
        )
        if not rule_fingerprints:
            return

        error_messages = []
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
            error_messages = "\n".join(error_messages)
            raise InvalidRule(
                f"\nIncomplete rules found🚨\n\n{error_messages}\n"
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
                active_loop = state.get(ACTIVE_LOOP, {}).get(LOOP_NAME)
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
        # do not raise an error if StateMachinePolicy didn't predict anything for stories;
        # however for rules StateMachinePolicy should always predict an action
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
        self,
        tracker: TrackerWithCachedStates,
        domain: Domain,
        gold_action_name: Text,
    ) -> Tuple[Optional[Text], Optional[Text]]:
        predicted_action_name, prediction_source = self._predict_next_action(
            tracker, domain
        )
        # if there is an active_loop,
        # StateMachinePolicy will always predict active_loop first,
        # but inside loop unhappy path there might be another action
        if (
            tracker.active_loop_name
            and predicted_action_name != gold_action_name
            and predicted_action_name == tracker.active_loop_name
        ):
            rasa.core.test.emulate_loop_rejection(tracker)
            (
                predicted_action_name,
                prediction_source,
            ) = self._predict_next_action(tracker, domain)

        return predicted_action_name, prediction_source

    def _collect_sources(
        self,
        tracker: TrackerWithCachedStates,
        predicted_action_name: Optional[Text],
        gold_action_name: Text,
        prediction_source: Optional[Text],
    ) -> None:
        # we need to remember which action should be predicted by the rule
        # in order to correctly output the names of the contradicting rules
        rule_name = tracker.sender_id
        if prediction_source.startswith(DEFAULT_RULES) or prediction_source.startswith(
            LOOP_RULES
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
        if not predicted_action_name or predicted_action_name == gold_action_name:
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
    ) -> Tuple[List[Text], Set[Text]]:
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
                (
                    predicted_action_name,
                    prediction_source,
                ) = self._predicted_action_name(
                    running_tracker, domain, gold_action_name
                )
                if collect_sources:
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
    ) -> Tuple[List[Text], Set[Text]]:
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
            interpreter: Interpreter which can be used by the polices for featurization.

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
            error_messages = "\n".join(error_messages)
            raise InvalidRule(
                f"\nContradicting rules or stories found 🚨\n\n{error_messages}\n"
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
        ) = self.featurizer.training_states_and_actions(
            rule_trackers, domain, omit_unset_slots=True
        )

        rules_lookup = self._create_lookup_from_states(
            rule_trackers_as_states, rule_trackers_as_actions
        )
        self.lookup[RULES] = self._remove_rule_snippet_predictions(rules_lookup)

        (
            story_trackers_as_states,
            story_trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(story_trackers, domain)

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
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers.

        Args:
            training_trackers: The list of the trackers.
            domain: The domain.
            interpreter: Interpreter which can be used by the polices for featurization.
        """
        # only consider original trackers (no augmented ones)
        training_trackers = [
            t for t in training_trackers if not getattr(t, "is_augmented", False)
        ]
        # trackers from rule-based training data
        rule_trackers = [t for t in training_trackers if t.is_rule_tracker]
        if self._restrict_rules:
            self._check_rule_restriction(rule_trackers)
        if self._check_for_contradictions:
            self._check_for_incomplete_rules(rule_trackers, domain)

        # trackers from ML-based training data
        story_trackers = [t for t in training_trackers if not t.is_rule_tracker]

        self._create_lookup_from_trackers(rule_trackers, story_trackers, domain)

        # # make this configurable because checking might take a lot of time
        # if self._check_for_contradictions:
        #     # using trackers here might not be the most efficient way, however
        #     # it allows us to directly test `predict_action_probabilities` method
        #     self.lookup[RULES_NOT_IN_STORIES] = self._analyze_rules(
        #         rule_trackers, training_trackers, domain
        #     )

        logger.debug(f"Memorized '{len(self.lookup[RULES])}' unique rules.")

    @staticmethod
    def _find_action_from_default_actions(
        tracker: DialogueStateTracker,
    ) -> Tuple[Optional[Text], Optional[Text]]:
        if (
            not tracker.latest_action_name == ACTION_LISTEN_NAME
            or not tracker.latest_message
        ):
            return None, None

        default_action_name = DEFAULT_ACTION_MAPPINGS.get(
            tracker.latest_message.intent.get(INTENT_NAME_KEY)
        )

        if default_action_name:
            logger.debug(f"Predicted default action '{default_action_name}'.")
            return (
                default_action_name,
                # create prediction source that corresponds to one of
                # default prediction sources in `_default_sources()`
                DEFAULT_RULES + tracker.latest_message.intent.get(INTENT_NAME_KEY),
            )

        return None, None

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> "PolicyPrediction":
        """Predicts the next action (see parent class for more information)."""
        prediction, _ = self._predict(tracker, domain)
        return prediction

    def _predict(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> Tuple[PolicyPrediction, Text]:
        action_name: str
        if tracker.latest_action_name == tracker.active_loop_name:
            action_name = ACTION_LISTEN_NAME
        else:
            action_name = "action_state_machine_action"

        return (
            self._rule_prediction(
                self._prediction_result(action_name, tracker, domain),
                None,
                returning_from_unhappy_path=False,
                is_end_to_end_prediction=True,
            ),
            None,
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
            result[
                domain.index_for_action(self._fallback_action_name)
            ] = self._core_fallback_threshold
        return result

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "lookup": self.lookup,
            "core_fallback_threshold": self._core_fallback_threshold,
            "core_fallback_action_name": self._fallback_action_name,
            "enable_fallback_prediction": self._enable_fallback_prediction,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "rule_policy.json"

    def get_rule_only_data(self) -> Dict[Text, Any]:
        """Gets the slots and loops that are used only in rule data.

        Returns:
            Slots and loops that are used only in rule data.
        """
        return {
            key: self.lookup.get(key, []) for key in [RULE_ONLY_SLOTS, RULE_ONLY_LOOPS]
        }
