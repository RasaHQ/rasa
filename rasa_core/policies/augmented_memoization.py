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
        augmented = [list(states)]
        augmented_states = list(states)
        for i in range(self.max_history - 1):
            augmented_states[i] = None
            augmented.append(list(augmented_states))
        return augmented

    def _back_to_the_future(self, tracker):
        mcfly_trackers = []
        if self.max_history > 1:
            historic_events = [[]]
            i = 0
            e_i_last = len(tracker._applied_events()) - 1
            for e_i, event in enumerate(
                    reversed(tracker._applied_events())):
                historic_events[i].append(event)

                if isinstance(event, ActionExecuted):
                    historic_events[i].reverse()
                    i += 1
                    if i == self.max_history - 1:
                        # we need i to be one less than max_history
                        # not to recall again with the same features
                        break
                    if e_i == e_i_last:
                        # if we arrived at the end of the tracker,
                        # the last historic_events repeat the tracker
                        # so we delete them
                        del historic_events[-1]
                        break
                    historic_events.append(historic_events[i - 1][::-1])

            historic_events.reverse()

            for events in historic_events:
                mcfly_tracker = tracker._init_copy()
                for e in events:
                    mcfly_tracker.update(e)
                mcfly_trackers.append(mcfly_tracker)

        return mcfly_trackers

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions.
        If memorized action was found returns 1.0 for its index,
        else returns 0.0 for all actions."""
        result = [0.0] * domain.num_actions

        if self.is_enabled:
            tracker_as_states = self.featurizer.prediction_states(
                                    [tracker], domain)
            states = tracker_as_states[0]
            logger.debug("Current tracker state {}".format(states))
            memorised = self._recall(states)
            if memorised is not None:
                logger.debug("Used memorised next action '{}'"
                             "".format(memorised))
                result[memorised] = 1.0
                return result

            # correctly forgetting slots
            logger.debug("Launch DeLorean...")
            mcfly_trackers = self._back_to_the_future(tracker)

            tracker_as_states = self.featurizer.prediction_states(
                                    mcfly_trackers, domain)
            for states in tracker_as_states:
                logger.debug("Current tracker state {}".format(states))
                memorised = self._recall(states)
                if memorised is not None:
                    logger.debug("Used memorised next action '{}'"
                                 "".format(memorised))
                    result[memorised] = 1.0
                    return result

        return result
