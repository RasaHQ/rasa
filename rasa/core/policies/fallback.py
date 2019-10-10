import json
import logging
import os
from typing import Any, List, Text, Optional, Dict, Tuple

from rasa.core.actions.action import ACTION_LISTEN_NAME

import rasa.utils.io

from rasa.core import utils
from rasa.core.domain import Domain
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.constants import FALLBACK_POLICY_PRIORITY

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
        priority: int = FALLBACK_POLICY_PRIORITY,
        nlu_threshold: float = 0.3,
        ambiguity_threshold: float = 0.1,
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
            ambiguity_threshold: threshold for minimum difference
                between confidences of the top two predictions
            fallback_action_name: name of the action to execute as a fallback
        """
        super(FallbackPolicy, self).__init__(priority=priority)

        self.nlu_threshold = nlu_threshold
        self.ambiguity_threshold = ambiguity_threshold
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

    def nlu_confidence_below_threshold(
        self, nlu_data: Dict[Text, Any]
    ) -> Tuple[bool, float]:
        """Check if the highest confidence is lower than ``nlu_threshold``."""

        # if NLU interpreter does not provide confidence score,
        # it is set to 1.0 here in order
        # to not override standard behaviour
        nlu_confidence = nlu_data.get("intent", {}).get("confidence", 1.0)
        return (nlu_confidence < self.nlu_threshold, nlu_confidence)

    def nlu_prediction_ambiguous(
        self, nlu_data: Dict[Text, Any]
    ) -> Tuple[bool, Optional[float]]:
        """Check if top 2 confidences are closer than ``ambiguity_threshold``."""
        intents = nlu_data.get("intent_ranking", [])
        if len(intents) >= 2:
            first_confidence = intents[0].get("confidence", 1.0)
            second_confidence = intents[1].get("confidence", 1.0)
            difference = first_confidence - second_confidence
            return (difference < self.ambiguity_threshold, difference)
        return (False, None)

    def should_nlu_fallback(
        self, nlu_data: Dict[Text, Any], last_action_name: Text
    ) -> bool:
        """Check if fallback action should be predicted.

        Checks for:
        - predicted NLU confidence is lower than ``nlu_threshold``
        - difference in top 2 NLU confidences lower than ``ambiguity_threshold``
        - last action is action listen
        """
        if last_action_name != ACTION_LISTEN_NAME:
            return False

        below_threshold, nlu_confidence = self.nlu_confidence_below_threshold(nlu_data)
        ambiguous_prediction, confidence_delta = self.nlu_prediction_ambiguous(nlu_data)

        if below_threshold:
            logger.debug(
                "NLU confidence {} is lower "
                "than NLU threshold {:.2f}. "
                "".format(nlu_confidence, self.nlu_threshold)
            )
            return True
        elif ambiguous_prediction:
            logger.debug(
                "The difference in NLU confidences "
                "for the top two intents ({}) is lower than "
                "the ambiguity threshold {:.2f}. "
                "".format(confidence_delta, self.ambiguity_threshold)
            )
            return True

        return False

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

        if tracker.latest_action_name == self.fallback_action_name:
            result = [0.0] * domain.num_actions
            idx = domain.index_for_action(ACTION_LISTEN_NAME)
            result[idx] = 1.0

        elif self.should_nlu_fallback(nlu_data, tracker.latest_action_name):
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
            "ambiguity_threshold": self.ambiguity_threshold,
            "core_threshold": self.core_threshold,
            "fallback_action_name": self.fallback_action_name,
        }
        rasa.utils.io.create_directory_for_file(config_file)
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path: Text) -> "FallbackPolicy":
        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "fallback_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(rasa.utils.io.read_file(meta_path))

        return cls(**meta)
