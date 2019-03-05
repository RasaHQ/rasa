import logging
import os
from typing import Any, List, Text, Optional

from rasa_core.actions.action import (ACTION_LISTEN_NAME, ACTION_RESTART_NAME,
                                      ACTION_BACK_NAME)

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.policies.policy import Policy
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class MappingPolicy(Policy):
    """Policy which maps intents directly to actions.

    Intents can be assigned actions in the domain file which are to be
    executed whenever the intent is detected. This policy takes precedence over
    any other policy."""

    def __init__(self, priority: Optional[int] = 5) -> None:
        """Create a new Mapping policy."""

        super(MappingPolicy, self).__init__(priority=priority)

    def train(self, *args, **kwargs) -> None:
        """Does nothing. This policy is deterministic."""

        pass

    def predict_action_probabilities(self, tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predicts the assigned action.

        If the current intent is assigned to an action that action will be
        predicted with the highest probability of all policies. If it is not
        the policy will predict zero for every action."""

        prediction = [0.0] * domain.num_actions
        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            intent = tracker.latest_message.intent.get('name')
            action = domain.intent_properties.get(intent, {}).get('triggers')
            if action:
                idx = domain.index_for_action(action)
                if idx is None:
                    logger.warning("MappingPolicy tried to predict unkown "
                                   "action '{}'.".format(action))
                else:
                    prediction[idx] = 1
            elif tracker.latest_message.intent.get('name') == 'restart':
                idx = domain.index_for_action(ACTION_RESTART_NAME)
                prediction[idx] = 1
            elif tracker.latest_message.intent.get('name') == 'back':
                idx = domain.index_for_action(ACTION_BACK_NAME)
                prediction[idx] = 1
        elif tracker.latest_action_name == ACTION_RESTART_NAME:
            idx = domain.index_for_action(ACTION_BACK_NAME)
            prediction[idx] = 1
        return prediction

    def persist(self, *args) -> None:
        """Does nothing since there is no data to be saved."""

        pass

    @classmethod
    def load(cls, *args) -> 'MappingPolicy':
        """Just returns the class since there is no data to be loaded."""

        return cls()
