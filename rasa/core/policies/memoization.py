import zlib

import base64
import json
import logging
import os
from tqdm import tqdm
from typing import Optional, Any, Dict, List, Text

import rasa.utils.io

from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, Event
from rasa.core.featurizers import TrackerFeaturizer, MaxHistoryTrackerFeaturizer
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled
from rasa.core.constants import MEMOIZATION_POLICY_PRIORITY

logger = logging.getLogger(__name__)


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

    SUPPORTS_ONLINE_TRAINING = True

    USE_NLU_CONFIDENCE_AS_SCORE = False

    @staticmethod
    def _standard_featurizer(
        max_history: Optional[int] = None,
    ) -> MaxHistoryTrackerFeaturizer:
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(
            state_featurizer=None,
            max_history=max_history,
            use_intent_probabilities=False,
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = MEMOIZATION_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        lookup: Optional[Dict] = None,
    ) -> None:

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority)

        self.max_history = self.featurizer.max_history
        self.lookup = lookup if lookup is not None else {}
        self.is_enabled = True

    def toggle(self, activate: bool) -> None:
        self.is_enabled = activate

    def _add_states_to_lookup(
        self, trackers_as_states, trackers_as_actions, domain, online=False
    ) -> None:
        """Add states to lookup dict"""
        if not trackers_as_states:
            return

        assert len(trackers_as_states[0]) == self.max_history, (
            "Trying to mem featurized data with {} historic turns. Expected: "
            "{}".format(len(trackers_as_states[0]), self.max_history)
        )

        assert len(trackers_as_actions[0]) == 1, (
            "The second dimension of trackers_as_action should be 1, "
            "instead of {}".format(len(trackers_as_actions[0]))
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
            feature_item = domain.index_for_action(action)

            if feature_key not in ambiguous_feature_keys:
                if feature_key in self.lookup.keys():
                    if self.lookup[feature_key] != feature_item:
                        if online:
                            logger.info(
                                "Original stories are "
                                "different for {} -- {}\n"
                                "Memorized the new ones for "
                                "now. Delete contradicting "
                                "examples after exporting "
                                "the new stories."
                                "".format(states, action)
                            )
                            self.lookup[feature_key] = feature_item
                        else:
                            # delete contradicting example created by
                            # partial history augmentation from memory
                            ambiguous_feature_keys.add(feature_key)
                            del self.lookup[feature_key]
                else:
                    self.lookup[feature_key] = feature_item
            pbar.set_postfix({"# examples": "{:d}".format(len(self.lookup))})

    def _create_feature_key(self, states: List[Dict]) -> Text:
        from rasa.utils import io

        feature_str = json.dumps(states, sort_keys=True).replace('"', "")
        if self.ENABLE_FEATURE_STRING_COMPRESSION:
            compressed = zlib.compress(bytes(feature_str, io.DEFAULT_ENCODING))
            return base64.b64encode(compressed).decode(io.DEFAULT_ENCODING)
        else:
            return feature_str

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers."""
        self.lookup = {}
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
        self._add_states_to_lookup(trackers_as_states, trackers_as_actions, domain)
        logger.debug("Memorized {} unique examples.".format(len(self.lookup)))

    def _recall_states(self, states: List[Dict[Text, float]]) -> Optional[int]:

        return self.lookup.get(self._create_feature_key(states))

    def recall(
        self,
        states: List[Dict[Text, float]],
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> Optional[int]:
        return self._recall_states(states)

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the next action the bot should take after seeing the tracker.

        Returns the list of probabilities for the next actions.
        If memorized action was found returns 1 for its index,
        else returns 0 for all actions.
        """
        result = self._default_predictions(domain)

        if not self.is_enabled:
            return result

        tracker_as_states = self.featurizer.prediction_states([tracker], domain)
        states = tracker_as_states[0]
        logger.debug(f"Current tracker state {states}")
        recalled = self.recall(states, tracker, domain)
        if recalled is not None:
            logger.debug(
                f"There is a memorised next action '{domain.action_names[recalled]}'"
            )

            if self.USE_NLU_CONFIDENCE_AS_SCORE:
                # the memoization will use the confidence of NLU on the latest
                # user message to set the confidence of the action
                score = tracker.latest_message.intent.get("confidence", 1.0)
            else:
                score = 1.0

            result[recalled] = score
        else:
            logger.debug("There is no memorised next action")

        return result

    def persist(self, path: Text) -> None:

        self.featurizer.persist(path)

        memorized_file = os.path.join(path, "memorized_turns.json")
        data = {
            "priority": self.priority,
            "max_history": self.max_history,
            "lookup": self.lookup,
        }
        rasa.utils.io.create_directory_for_file(memorized_file)
        rasa.utils.io.dump_obj_as_json_to_file(memorized_file, data)

    @classmethod
    def load(cls, path: Text) -> "MemoizationPolicy":

        featurizer = TrackerFeaturizer.load(path)
        memorized_file = os.path.join(path, "memorized_turns.json")
        if os.path.isfile(memorized_file):
            data = json.loads(rasa.utils.io.read_file(memorized_file))
            return cls(
                featurizer=featurizer, priority=data["priority"], lookup=data["lookup"]
            )
        else:
            logger.info(
                "Couldn't load memoization for policy. "
                "File '{}' doesn't exist. Falling back to empty "
                "turn memory.".format(memorized_file)
            )
            return cls()


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
        tracker: DialogueStateTracker,
        max_num_events: Optional[int] = None,
        again: bool = False,
    ) -> Optional[DialogueStateTracker]:
        """Send Marty to the past to get
        the new featurization for the future"""

        idx_of_first_action = None
        idx_of_second_action = None

        if max_num_events:
            applied_events = tracker.applied_events()[-max_num_events:]
        else:
            applied_events = tracker.applied_events()

        # we need to find second executed action
        for e_i, event in enumerate(applied_events):
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
        events = applied_events[idx_to_use:]
        if not events:
            return

        mcfly_tracker = tracker.init_copy()
        for e in events:
            mcfly_tracker.update(e)

        return mcfly_tracker

    def _recall_using_delorean(
        self, old_states: List, tracker: DialogueStateTracker, domain: Domain,
    ) -> Optional[int]:
        """Applies to the future idea to change the past and get the new future.

        Recursively go to the past to correctly forget slots,
        and then back to the future to recall.

        Args:
            old_states: List of states.
            tracker: The tracker.
            domain: The Domain.

        Returns:
            The name of the action.
        """
        logger.debug("Launch DeLorean...")

        # Count how many events we need to look at, based on `max_history`
        max_num_events = num_past_events_from_max_history(tracker, self.max_history)

        mcfly_tracker = self._back_to_the_future(tracker, max_num_events=max_num_events)
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
        self, states: List, tracker: DialogueStateTracker, domain: Domain,
    ) -> Optional[int]:
        """Finds the action based on the given states.

        Uses back to the future idea to change the past and check whether the new future
        can be used to recall the action.

        Args:
            states: List of states.
            tracker: The tracker.
            domain: The Domain.

        Returns:
            The index of the action.
        """
        predicted_action_index = self._recall_states(states)
        if predicted_action_index is None:
            # let's try a different method to recall that tracker
            return self._recall_using_delorean(states, tracker, domain)
        else:
            return predicted_action_index


def num_past_events_from_max_history(
    tracker: DialogueStateTracker,
    max_history: Optional[int],
) -> Optional[int]:
    """Computes the number of events in the tracker that correspond to max_history.

    Args:
        tracker: Some tracker holding the events
        max_history: The number of actions to count

    Returns:
        The number of actions, as counted from the end of the event list, that should
        be taken into accout according to the `max_history` setting.
    """
    if not max_history:
        return None
    num_events = 0
    num_actions = 0
    for event in reversed(tracker.applied_events()):
        num_events += 1
        if isinstance(event, ActionExecuted):
            num_actions += 1
        if num_actions > max_history:
            break
    return num_events