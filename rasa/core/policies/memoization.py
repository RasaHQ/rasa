import zlib

import base64
import json
import logging

from tqdm import tqdm
from typing import Optional, Any, Dict, List, Text

import rasa.utils.io
import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_POLICIES
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.utils.io import is_logging_disabled
from rasa.core.constants import MEMOIZATION_POLICY_PRIORITY

logger = logging.getLogger(__name__)

# temporary constants to support back compatibility
MAX_HISTORY_NOT_SET = -1
OLD_DEFAULT_MAX_HISTORY = 5


class MemoizationPolicy(Policy):
    """The policy that remembers exact examples of
    `max_history` turns from training stories.

    Since `slots` that are set some time in the past are
    preserved in all future feature vectors until they are set
    to None, this policy implicitly remembers and most importantly
    recalls examples in the context of the current dialogue
    longer than `max_history`.

    This policy is not supposed to be the only policy in an ensemble,
    it is optimized for precision and not recall.
    It should get a 100% precision because it emits probabilities of 1.1
    along it's predictions, which makes every mistake fatal as
    no other policy can overrule it.

    If it is needed to recall turns from training dialogues where
    some slots might not be set during prediction time, and there are
    training stories for this, use AugmentedMemoizationPolicy.
    """

    ENABLE_FEATURE_STRING_COMPRESSION = True

    USE_NLU_CONFIDENCE_AS_SCORE = False

    @staticmethod
    def _standard_featurizer(
        max_history: Optional[int] = None,
    ) -> MaxHistoryTrackerFeaturizer:
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(
            state_featurizer=None, max_history=max_history
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = MEMOIZATION_POLICY_PRIORITY,
        max_history: Optional[int] = MAX_HISTORY_NOT_SET,
        lookup: Optional[Dict] = None,
    ) -> None:
        """Initialize the policy.

        Args:
            featurizer: tracker featurizer
            priority: the priority of the policy
            max_history: maximum history to take into account when featurizing trackers
            lookup: a dictionary that stores featurized tracker states and
                predicted actions for them
        """

        if max_history == MAX_HISTORY_NOT_SET:
            max_history = OLD_DEFAULT_MAX_HISTORY  # old default value
            rasa.shared.utils.io.raise_warning(
                f"Please configure the max history in your configuration file, "
                f"currently 'max_history' is set to old default value of "
                f"'{max_history}'. If you want to have infinite max history "
                f"set it to 'None' explicitly. We will change the default value of "
                f"'max_history' in the future to 'None'.",
                DeprecationWarning,
                docs=DOCS_URL_POLICIES,
            )

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority)

        self.max_history = self.featurizer.max_history
        self.lookup = lookup if lookup is not None else {}

    def _create_lookup_from_states(
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

        if not trackers_as_states:
            return lookup

        assert len(trackers_as_actions[0]) == 1, (
            f"The second dimension of trackers_as_action should be 1, "
            f"instead of {len(trackers_as_actions[0])}"
        )

        ambiguous_feature_keys = set()

        pbar = tqdm(
            zip(trackers_as_states, trackers_as_actions),
            desc="Processed actions",
            disable=is_logging_disabled(),
        )
        for states, actions in pbar:
            action = actions[0]

            feature_key = self._create_feature_key(states)
            if not feature_key:
                continue

            if feature_key not in ambiguous_feature_keys:
                if feature_key in lookup.keys():
                    if lookup[feature_key] != action:
                        # delete contradicting example created by
                        # partial history augmentation from memory
                        ambiguous_feature_keys.add(feature_key)
                        del lookup[feature_key]
                else:
                    lookup[feature_key] = action
            pbar.set_postfix({"# examples": "{:d}".format(len(lookup))})

        return lookup

    def _create_feature_key(self, states: List[State]) -> Text:
        # we sort keys to make sure that the same states
        # represented as dictionaries have the same json strings
        # quotes are removed for aesthetic reasons
        feature_str = json.dumps(states, sort_keys=True).replace('"', "")
        if self.ENABLE_FEATURE_STRING_COMPRESSION:
            compressed = zlib.compress(
                bytes(feature_str, rasa.shared.utils.io.DEFAULT_ENCODING)
            )
            return base64.b64encode(compressed).decode(
                rasa.shared.utils.io.DEFAULT_ENCODING
            )
        else:
            return feature_str

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        # only considers original trackers (no augmented ones)
        training_trackers = [
            t
            for t in training_trackers
            if not hasattr(t, "is_augmented") or not t.is_augmented
        ]
        (
            trackers_as_states,
            trackers_as_actions,
        ) = self.featurizer.training_states_and_actions(training_trackers, domain)
        self.lookup = self._create_lookup_from_states(
            trackers_as_states, trackers_as_actions
        )
        logger.debug(f"Memorized {len(self.lookup)} unique examples.")

    def _recall_states(self, states: List[State]) -> Optional[Text]:
        return self.lookup.get(self._create_feature_key(states))

    def recall(
        self, states: List[State], tracker: DialogueStateTracker, domain: Domain
    ) -> Optional[Text]:
        return self._recall_states(states)

    def _prediction_result(
        self, action_name: Text, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        result = self._default_predictions(domain)
        if action_name:
            if self.USE_NLU_CONFIDENCE_AS_SCORE:
                # the memoization will use the confidence of NLU on the latest
                # user message to set the confidence of the action
                score = tracker.latest_message.intent.get("confidence", 1.0)
            else:
                score = 1.0

            result[domain.index_for_action(action_name)] = score

        return result

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        result = self._default_predictions(domain)

        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]
        logger.debug(f"Current tracker state:{self.format_tracker_states(states)}")
        predicted_action_name = self.recall(states, tracker, domain)
        if predicted_action_name is not None:
            logger.debug(f"There is a memorised next action '{predicted_action_name}'")
            result = self._prediction_result(predicted_action_name, tracker, domain)
        else:
            logger.debug("There is no memorised next action")

        return self._prediction(result)

    def _metadata(self) -> Dict[Text, Any]:
        return {
            "priority": self.priority,
            "max_history": self.max_history,
            "lookup": self.lookup,
        }

    @classmethod
    def _metadata_filename(cls) -> Text:
        return "memorized_turns.json"


class AugmentedMemoizationPolicy(MemoizationPolicy):
    """The policy that remembers examples from training stories
    for `max_history` turns.

    If it is needed to recall turns from training dialogues
    where some slots might not be set during prediction time,
    add relevant stories without such slots to training data.
    E.g. reminder stories.

    Since `slots` that are set some time in the past are
    preserved in all future feature vectors until they are set
    to None, this policy has a capability to recall the turns
    up to `max_history` from training stories during prediction
    even if additional slots were filled in the past
    for current dialogue.
    """

    @staticmethod
    def _back_to_the_future(
        tracker, again: bool = False
    ) -> Optional[DialogueStateTracker]:
        """Send Marty to the past to get
        the new featurization for the future"""

        idx_of_first_action = None
        idx_of_second_action = None

        # we need to find second executed action
        for e_i, event in enumerate(tracker.applied_events()):
            # find second ActionExecuted
            if isinstance(event, ActionExecuted):
                if idx_of_first_action is None:
                    idx_of_first_action = e_i
                else:
                    idx_of_second_action = e_i
                    break

        # use first action, if we went first time and second action, if we went again
        idx_to_use = idx_of_second_action if again else idx_of_first_action
        if idx_to_use is None:
            return

        # make second ActionExecuted the first one
        events = tracker.applied_events()[idx_to_use:]
        if not events:
            return

        mcfly_tracker = tracker.init_copy()
        for e in events:
            mcfly_tracker.update(e)

        return mcfly_tracker

    def _recall_using_delorean(self, old_states, tracker, domain) -> Optional[Text]:
        """Recursively go to the past to correctly forget slots,
        and then back to the future to recall."""

        logger.debug("Launch DeLorean...")

        mcfly_tracker = self._back_to_the_future(tracker)
        while mcfly_tracker is not None:
            tracker_as_states = self.featurizer.prediction_states(
                [mcfly_tracker], domain
            )
            states = tracker_as_states[0]

            if old_states != states:
                # check if we like new futures
                memorised = self._recall_states(states)
                if memorised is not None:
                    logger.debug(f"Current tracker state {states}")
                    return memorised
                old_states = states

            # go back again
            mcfly_tracker = self._back_to_the_future(mcfly_tracker, again=True)

        # No match found
        logger.debug(f"Current tracker state {old_states}")
        return None

    def recall(
        self, states: List[State], tracker: DialogueStateTracker, domain: Domain
    ) -> Optional[Text]:

        predicted_action_name = self._recall_states(states)
        if predicted_action_name is None:
            # let's try a different method to recall that tracker
            return self._recall_using_delorean(states, tracker, domain)
        else:
            return predicted_action_name
