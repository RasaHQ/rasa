import logging
import json
import os
from tqdm import tqdm
import typing
from typing import Any, List, Text, Optional, Dict

from rasa.constants import DOCS_URL_POLICIES
import rasa.utils.io

from rasa.core.actions.action import (
    ACTION_BACK_NAME,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
)
from rasa.core.constants import (
    USER_INTENT_BACK,
    USER_INTENT_RESTART,
    USER_INTENT_SESSION_START,
)
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.events import ActionExecuted
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.featurizers import TrackerFeaturizer, MaxHistoryTrackerFeaturizer
from rasa.core.constants import MAPPING_POLICY_PRIORITY
from rasa.utils.common import raise_warning

if typing.TYPE_CHECKING:
    from rasa.core.policies.ensemble import PolicyEnsemble


logger = logging.getLogger(__name__)


class InferredMappingPolicy(Policy):
    """Policy which maps intents directly to actions.

    Intents are assigned to actions if an intent is *always* followed by the 
    same action in all stories.
    """

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

    def __init__(self,
                 featurizer: Optional[TrackerFeaturizer] = None,                 
                 priority: int = MAPPING_POLICY_PRIORITY,
                 max_history: Optional[int] = None,                 
                 lookup: Optional[Dict] = None,) -> None:
        """Create a new Mapping policy."""

        max_history = 2
        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority=priority)

        self.max_history = self.featurizer.max_history
        self.lookup = lookup if lookup is not None else {}        

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
        non_mapping_intents = set()
        
        pbar = tqdm(
            zip(trackers_as_states, trackers_as_actions),
            desc="Processed actions",
            disable=False,
        )
        for states, actions in pbar:
            action = actions[0]
            intent = None
            try:
                if "prev_action_listen" in states[-1].keys():
                    intent_key = [k for k in states[-1].keys() if k.startswith("intent")][0]
                    intent = intent_key.split("intent_")[1]
            except:
                pass

            feature_item = action # domain.index_for_action(action)

            if intent in self.lookup.keys():
                if self.lookup[intent] != feature_item:
                    non_mapping_intents.add(intent)
            elif intent:
                self.lookup[intent] = feature_item
            pbar.set_postfix({"# examples": "{:d}".format(len(self.lookup))})

        for intent in non_mapping_intents:
            del self.lookup[intent]
        logger.debug(f"the following intents are contextual, and hence not mapped: {non_mapping_intents}")
        logger.debug(f"the following intents are always mapped to the same action: {self.lookup.keys()}")

        
    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Learn which intents to map to a fixed action."""
        self.lookup = {}
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
        #memorize some shit
        pass

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the assigned action.

        If the current intent is assigned to an action that action will be
        predicted with the highest probability of all policies. If it is not
        the policy will predict zero for every action."""

        result = self._default_predictions(domain)        

        intent = tracker.latest_message.intent.get("name")
        if intent == USER_INTENT_RESTART:
            action = ACTION_RESTART_NAME
        elif intent == USER_INTENT_BACK:
            action = ACTION_BACK_NAME
        elif intent == USER_INTENT_SESSION_START:
            action = ACTION_SESSION_START_NAME
        else:
            action = self.lookup.get(intent)

        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            # predict mapped action
            if action:
                idx = domain.index_for_action(action)
                if idx is None:
                    raise_warning(
                        f"MappingPolicy tried to predict unknown "
                        f"action '{action}'. Make sure all mapped actions are "
                        f"listed in the domain.",
                        docs=DOCS_URL_POLICIES + "#mapping-policy",
                    )
                else:
                    result[idx] = 1

            if any(result):
                logger.debug(
                    "The predicted intent '{}' is mapped to "
                    " action '{}' in the domain."
                    "".format(intent, action)
                )
        elif tracker.latest_action_name == action and action is not None:
            # predict next action_listen after mapped action
            latest_action = tracker.get_last_event_for(ActionExecuted)
            assert latest_action.action_name == action
            if latest_action.policy and latest_action.policy.endswith(
                type(self).__name__
            ):
                # this ensures that we only predict listen,
                # if we predicted the mapped action
                logger.debug(
                    "The mapped action, '{}', for this intent, '{}', was "
                    "executed last so MappingPolicy is returning to "
                    "action_listen.".format(action, intent)
                )

                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                result[idx] = 1
            else:
                logger.debug(
                    "The mapped action, '{}', for the intent, '{}', was "
                    "executed last, but it was predicted by another policy, '{}', "
                    "so MappingPolicy is not predicting any action.".format(
                        action, intent, latest_action.policy
                    )
                )
        elif action == ACTION_RESTART_NAME:
            logger.debug("Restarting the conversation with action_restart.")
            idx = domain.index_for_action(ACTION_RESTART_NAME)
            result[idx] = 1
        else:
            logger.debug(
                "There is no mapped action for the predicted intent, "
                "'{}'.".format(intent)
            )
        return result

    def persist(self, path: Text) -> None:
        """Only persists the priority."""

        config_file = os.path.join(path, "inferred_mapping_policy.json")
        data = {"priority": self.priority, "max_history": self.max_history, "lookup": self.lookup}
        rasa.utils.io.create_directory_for_file(config_file)
        rasa.utils.io.dump_obj_as_json_to_file(config_file, data)

    @classmethod
    def load(cls, path: Text) -> "MappingPolicy":
        """Returns the class with the configured priority."""

        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "inferred_mapping_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(rasa.utils.io.read_file(meta_path))

        return cls(**meta)
