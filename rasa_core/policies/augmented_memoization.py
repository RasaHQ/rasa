from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from typing import Optional, Any, Dict, List, Text

from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.events import ActionExecuted

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain

ENABLE_FEATURE_STRING_COMPRESSION = True


class AugmentedMemoizationPolicy(MemoizationPolicy):
    SUPPORTS_ONLINE_TRAINING = True

    def _preprocess_states(self, states):
        # type: (List[Dict[Text, float]]) -> List[List[Dict[Text, float]]]
        """Overrides the helper method to preprocess tracker's states.

        Creates a list of states with deleted history
        to add the ability of augmented memoization
        to recall partial history"""

        augmented = [list(states)]
        augmented_states = list(states)
        for i in range(self.max_history - 1):
            augmented_states[i] = None
            augmented.append(list(augmented_states))
        return augmented

    def _back_to_the_future(self, tracker):
        if self.max_history <= 1:
            return []

        historic_events = []
        collected_events = []

        idx_of_last_evt = len(tracker.applied_events()) - 1

        for e_i, event in enumerate(reversed(tracker.applied_events())):
            collected_events.append(event)

            if isinstance(event, ActionExecuted):
                if e_i == idx_of_last_evt:
                    # if arrived at the end of the tracker,
                    # the last historic_events repeat the tracker
                    # so `break` is called before appending them
                    break

                historic_events.append(collected_events[:])

                if len(historic_events) == self.max_history - 1:
                    # the length of `historic_events` should be
                    # one less than max_history, in order
                    # to not recall again with the same features
                    break

        mcfly_trackers = []
        for events in reversed(historic_events):
            mcfly_tracker = tracker.init_copy()
            for e in reversed(events):
                mcfly_tracker.update(e)
            mcfly_trackers.append(mcfly_tracker)

        return mcfly_trackers

    def _recall_using_delorean(self, tracker, domain):
        # correctly forgetting slots

        logger.debug("Launch DeLorean...")
        mcfly_trackers = self._back_to_the_future(tracker)

        tracker_as_states = self.featurizer.prediction_states(
                                mcfly_trackers, domain)

        for states in tracker_as_states:
            logger.debug("Current tracker state {}".format(states))
            memorised = self._recall(states)
            if memorised is not None:
                return memorised

        # No match found
        return None

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions.
        If memorized action was found returns 1.0 for its index,
        else returns 0.0 for all actions."""
        result = [0.0] * domain.num_actions

        if not self.is_enabled:
            return result

        tracker_as_states = self.featurizer.prediction_states(
                                [tracker], domain)
        states = tracker_as_states[0]
        logger.debug("Current tracker state {}".format(states))

        recalled = self._recall(states)
        if recalled is None:
            # let's try a different method to recall that tracker
            recalled = self._recall_using_delorean(tracker, domain)

        if recalled is not None:
            logger.debug("Used memorised next action '{}'"
                         "".format(recalled))
            result[recalled] = 1.0

        return result
