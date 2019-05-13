import json
import logging
import os
from typing import Any, List, Text

from rasa.core.actions.action import ACTION_LISTEN_NAME

import rasa.utils.io

from rasa.core import utils
from rasa.core.domain import Domain
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


class FallbackPolicy(Policy):
    """Policy which predicts fallback actions.

    A fallback can be triggered by a low confidence score on a
    NLU prediction or by a low confidence score on an action
    prediction. """

    @staticmethod
    def _standard_featurizer():
        return None

    def __init__(
        self,
        priority: int = 3,
        nlu_threshold: float = 0.3,
        core_threshold: float = 0.3,
        fallback_action_name: Text = "action_default_fallback",
    ) -> None:
        """Create a new Fallback policy.

        Args:
            core_threshold: if NLU confidence threshold is met,
                predict fallback action with confidence `core_threshold`.
                If this is the highest confidence in the ensemble,
                the fallback action will be executed.
            nlu_threshold: minimum threshold for NLU confidence.
                If intent prediction confidence is lower than this,
                predict fallback action with confidence 1.0.
            fallback_action_name: name of the action to execute as a fallback
        """
        super(FallbackPolicy, self).__init__(priority=priority)

        self.nlu_threshold = nlu_threshold
        self.core_threshold = core_threshold
        self.fallback_action_name = fallback_action_name

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any
    ) -> None:
        """Does nothing. This policy is deterministic."""

        pass

    def should_nlu_fallback(
        self, nlu_confidence: float, last_action_name: Text
    ) -> bool:
        """Checks if fallback action should be predicted.

        Checks for:
        - predicted NLU confidence is lower than ``nlu_threshold``
        - last action is action listen
        """

        return (
            nlu_confidence < self.nlu_threshold
            and last_action_name == ACTION_LISTEN_NAME
        )

    def fallback_scores(self, domain, fallback_score=1.0):
        """Prediction scores used if a fallback is necessary."""

        result = [0.0] * domain.num_actions
        idx = domain.index_for_action(self.fallback_action_name)
        result[idx] = fallback_score
        return result

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts a fallback action.

        The fallback action is predicted if the NLU confidence is low
        or no other policy has a high-confidence prediction.
        """

        nlu_data = tracker.latest_message.parse_data

        # if NLU interpreter does not provide confidence score,
        # it is set to 1.0 here in order
        # to not override standard behaviour
        nlu_confidence = nlu_data.get("intent", {}).get("confidence", 1.0)

        if tracker.latest_action_name == self.fallback_action_name:
            result = [0.0] * domain.num_actions
            idx = domain.index_for_action(ACTION_LISTEN_NAME)
            result[idx] = 1.0

        elif self.should_nlu_fallback(nlu_confidence, tracker.latest_action_name):
            logger.debug(
                "NLU confidence {} is lower "
                "than NLU threshold {}. "
                "".format(nlu_confidence, self.nlu_threshold)
            )
            result = self.fallback_scores(domain)

        else:
            # NLU confidence threshold is met, so
            # predict fallback action with confidence `core_threshold`
            # if this is the highest confidence in the ensemble,
            # the fallback action will be executed.
            logger.debug(
                "NLU confidence threshold met, confidence of "
                "fallback action set to core threshold ({}).".format(
                    self.core_threshold
                )
            )
            result = self.fallback_scores(domain, self.core_threshold)

        return result

    def persist(self, path: Text) -> None:
        """Persists the policy to storage."""

        config_file = os.path.join(path, "fallback_policy.json")
        meta = {
            "priority": self.priority,
            "nlu_threshold": self.nlu_threshold,
            "core_threshold": self.core_threshold,
            "fallback_action_name": self.fallback_action_name,
        }
        utils.create_dir_for_file(config_file)
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path: Text) -> "FallbackPolicy":
        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "fallback_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(rasa.utils.io.read_file(meta_path))

        return cls(**meta)
