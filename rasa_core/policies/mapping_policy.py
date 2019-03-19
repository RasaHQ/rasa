import logging
from typing import Any, List, Text

from rasa_core.actions.action import (
    ACTION_BACK_NAME, ACTION_LISTEN_NAME, ACTION_RESTART_NAME)
from rasa_core.constants import USER_INTENT_BACK, USER_INTENT_RESTART
from rasa_core.domain import Domain
from rasa_core.policies.policy import Policy
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class MappingPolicy(Policy):
    """Policy which maps intents directly to actions.

    Intents can be assigned actions in the domain file which are to be
    executed whenever the intent is detected. This policy takes precedence over
    any other policy."""

    def __init__(self, priority: int = 5) -> None:
        """Create a new Mapping policy."""

        super(MappingPolicy, self).__init__(priority=priority)

    def train(self,
              training_trackers: List[DialogueStateTracker],
              domain: Domain,
              **kwargs: Any
              ) -> None:
        """Does nothing. This policy is deterministic."""

        pass

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predicts the assigned action.

        If the current intent is assigned to an action that action will be
        predicted with the highest probability of all policies. If it is not
        the policy will predict zero for every action."""

        prediction = [0.0] * domain.num_actions
        intent = tracker.latest_message.intent.get('name')
        action = domain.intent_properties.get(intent, {}).get('triggers')
        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            if action:
                idx = domain.index_for_action(action)
                if idx is None:
                    logger.warning("MappingPolicy tried to predict unkown "
                                   "action '{}'.".format(action))
                else:
                    prediction[idx] = 1
            elif intent == USER_INTENT_RESTART:
                idx = domain.index_for_action(ACTION_RESTART_NAME)
                prediction[idx] = 1
            elif intent == USER_INTENT_BACK:
                idx = domain.index_for_action(ACTION_BACK_NAME)
                prediction[idx] = 1
        elif tracker.latest_action_name == action:
            idx = domain.index_for_action(ACTION_BACK_NAME)
            prediction[idx] = 1
        return prediction

    def persist(self, path: Text) -> None:
        """Does nothing since there is no data to be saved."""

        pass

    @classmethod
    def load(cls, path: Text) -> 'MappingPolicy':
        """Just returns the class since there is no data to be loaded."""

        return cls()
