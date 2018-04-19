from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import base64
import io
import json
import logging
import os
import zlib
import typing
from tqdm import tqdm

from builtins import bytes
from typing import Optional, Any, Dict, List, Text

from rasa_core.policies.policy import Policy
from rasa_core import utils
from rasa_core.featurizers import \
    TrackerFeaturizer, MaxHistoryTrackerFeaturizer
from rasa_core.events import ActionExecuted

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.domain import Domain

ENABLE_FEATURE_STRING_COMPRESSION = True


class MemoizationPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    @classmethod
    def _standard_featurizer(cls, max_history=None):
        max_history = max_history or cls.MAX_HISTORY_DEFAULT
        # Memoization policy always uses MaxHistoryTrackerFeaturizer
        # without state_featurizer
        return MaxHistoryTrackerFeaturizer(None, max_history)

    def __init__(self, max_history=None, lookup=None):
        # type: (int, Optional[Dict]) -> None

        featurizer = self._standard_featurizer(max_history)
        super(MemoizationPolicy, self).__init__(featurizer)

        self.max_history = self.featurizer.max_history
        self.lookup = lookup if lookup is not None else {}
        self.is_enabled = True

    def toggle(self, activate):
        # type: (bool) -> None
        self.is_enabled = activate

    def _create_partial_histories(self, states):
        augmented = [list(states)]
        augmented_states = list(states)
        for i in range(self.max_history - 1):
            augmented_states[i] = None
            augmented.append(list(augmented_states))
        return augmented

    def _add(self, trackers_as_states, trackers_as_actions,
             domain, online=False):

        if not trackers_as_states:
            return

        assert len(trackers_as_states[0]) == self.max_history, \
            ("Trying to mem featurized data with {} historic turns. Expected: "
             "{}".format(len(trackers_as_states[0]), self.max_history))

        assert len(trackers_as_actions[0]) == 1, \
            ("The second dimension of trackers_as_action should be 1, "
             "instead of {}".format(len(trackers_as_actions[0])))

        ambiguous_feature_keys = set()

        pbar = tqdm(zip(trackers_as_states, trackers_as_actions),
                    desc="Processed actions", disable=online)
        for states, actions in pbar:
            action = actions[0]
            for i, states_augmented in enumerate(
                    self._create_partial_histories(states)):
                feature_key = self._create_feature_key(states_augmented)
                feature_item = domain.index_for_action(action)

                if feature_key not in ambiguous_feature_keys:
                    if feature_key in self.lookup.keys():
                        if self.lookup[feature_key] != feature_item:
                            if online and i == 0:
                                logger.info("Original stories are "
                                            "different for {} -- {}\n"
                                            "Memorized the new ones for "
                                            "now. Delete contradicting "
                                            "examples after exporting "
                                            "the new stories."
                                            "".format(states_augmented,
                                                      action))
                                self.lookup[feature_key] = feature_item
                            else:
                                # delete contradicting example created by
                                # partial history augmentation from memory
                                ambiguous_feature_keys.add(feature_key)
                                del self.lookup[feature_key]
                    else:
                        self.lookup[feature_key] = feature_item
                pbar.set_postfix({"# examples": len(self.lookup)})

    @staticmethod
    def _create_feature_key(states):
        feature_str = json.dumps(states, sort_keys=True).replace("\"", "")
        if ENABLE_FEATURE_STRING_COMPRESSION:
            compressed = zlib.compress(bytes(feature_str, "utf-8"))
            return base64.b64encode(compressed).decode("utf-8")
        else:
            return feature_str

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> Dict[Text: Any]
        """Trains the policy on given training trackers."""
        self.lookup = {}

        (trackers_as_states,
         trackers_as_actions) = self.featurizer.training_states_and_actions(
                                    training_trackers, domain)
        self._add(trackers_as_states, trackers_as_actions, domain)
        logger.info("Memorized {} unique augmented examples."
                    "".format(len(self.lookup)))

    def continue_training(self, training_trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None

        # add only the last tracker, because it is the only new one
        (trackers_as_states,
         trackers_as_actions) = self.featurizer.training_states_and_actions(
                                    training_trackers[-1:], domain)
        self._add(trackers_as_states, trackers_as_actions,
                  domain, online=True)

    def _recall(self, states):
        for states_aug in self._create_partial_histories(states):
            memorised = self.lookup.get(
                self._create_feature_key(states_aug))
            if memorised is not None:
                return memorised
        return None

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

    def persist(self, path):
        # type: (Text) -> None

        super(MemoizationPolicy, self).persist(path)

        memorized_file = os.path.join(path, 'memorized_turns.json')
        data = {
            "lookup": self.lookup
        }
        utils.create_dir_for_file(memorized_file)
        utils.dump_obj_as_json_to_file(memorized_file, data)

    @classmethod
    def load(cls, path):
        # type: (Text) -> MemoizationPolicy

        featurizer = TrackerFeaturizer.load(path)
        assert isinstance(featurizer, MaxHistoryTrackerFeaturizer), \
            ("Loaded featurizer of type {}, should be "
             "MaxHistoryTrackerFeaturizer."
             "".format(type(featurizer).__name__))

        memorized_file = os.path.join(path, 'memorized_turns.json')
        if os.path.isfile(memorized_file):
            with io.open(memorized_file) as f:
                data = json.loads(f.read())
            return cls(featurizer.max_history, data["lookup"])
        else:
            logger.info("Couldn't load memoization for policy. "
                        "File '{}' doesn't exist. Falling back to empty "
                        "turn memory.".format(memorized_file))
            return MemoizationPolicy()
