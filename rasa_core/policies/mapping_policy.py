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

    Intents can be assigned actioins in the domain file which are to be
    executed whenever the intent is detected. This policy takes precedence over
    any other policy."""

    def __init__(self) -> None:
        """Create a new Mapping policy.

        Args:
            domain: the domain object
        """

        super(MappingPolicy, self).__init__()
        self.last_action_name = None

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
        the policy will predict zero for every action.
        """

        current_intent = tracker.latest_message.intent['name']
        action_name = domain.intent_properties[current_intent].get('maps_to')

        prediction = [0.0] * domain.num_actions
        if action_name is not None:
            if self.last_action_name == action_name:
                idx = domain.index_for_action(ACTION_LISTEN_NAME)
                prediction[idx] = MAPPING_SCORE
            else:
                idx = domain.index_for_action(action_name)
                prediction[idx] = MAPPING_SCORE
                self.last_action_name = action_name
        return prediction

    def persist(self, *args) -> None:
        """Does nothing since there is no data to be saved."""

        pass

    @classmethod
    def load(cls, *args) -> 'MappingPolicy':
        """Just returns the class since there is no data to be loaded."""

        return cls()
