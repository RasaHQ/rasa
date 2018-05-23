from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from typing import Dict, List, Text, Optional

from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.events import ActionExecuted

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain


class AugmentedMemoizationPolicy(MemoizationPolicy):
    """The policy that remembers examples from training stories
        for up to `max_history` turns.

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
            memorised = self._recall_states(states)
            if memorised is not None:
                return memorised

        # No match found
        return None

    def recall(self,
               states,  # type: List[Dict[Text, float]]
               tracker,  # type: DialogueStateTracker
               domain  # type: Domain
               ):
        # type: (...) -> Optional[int]

        recalled = self._recall_states(states)
        if recalled is None:
            # let's try a different method to recall that tracker
            return self._recall_using_delorean(tracker, domain)
        else:
            return recalled
