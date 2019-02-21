import logging
import os
from typing import Any, List, Text

from rasa_core.actions.action import ACTION_LISTEN_NAME

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.policies.policy import Policy
from rasa_core.trackers import DialogueStateTracker
from rasa_core.constants import MAPPING_SCORE

logger = logging.getLogger(__name__)


class MappingPolicy(Policy):
    """Policy which maps intents directly to actions.

    Intents can be assigned actions in the domain file which are to be
    executed whenever the intent is detected. This policy takes precedence over
    any other policy."""

    def __init__(self) -> None:
        """Create a new Mapping policy."""

        super(MappingPolicy, self).__init__()
        self.last_msg_id = None

    def train(self, *args, **kwargs) -> None:
        """Does nothing. This policy is deterministic."""

        pass

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predicts the assigned action.

        If the current intent is assigned to an action that action will be
        predicted with the highest probability of all policies. If it is not
        the policy will predict zero for every action."""

        intent = tracker.latest_message.intent.get('name')
        msg_id = tracker.latest_message.message_id
        action_name = domain.intent_properties.get(intent, {}).get('triggers')

        prediction = [0.0] * domain.num_actions
        if msg_id != self.last_msg_id and action_name is not None:
            idx = domain.index_for_action(action_name)
            if idx is None:
                logger.warning("MappingPolicy tried to predict unkown action "
                               "'{}'".format(action_name))
            else:
                prediction[idx] = MAPPING_SCORE
            self.last_msg_id = msg_id
        return prediction

    def persist(self, *args) -> None:
        """Does nothing since there is no data to be saved."""

        pass

    @classmethod
    def load(cls, *args) -> 'MappingPolicy':
        """Just returns the class since there is no data to be loaded."""

        return cls()
