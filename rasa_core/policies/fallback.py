from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import json
import io

from typing import Any
from typing import List
from typing import Optional
from typing import Text

from rasa_core import utils
from rasa_core.domain import Domain
from rasa_core.policies.policy import Policy
from rasa_core.featurizers import Featurizer
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.data import DialogueTrainingData

logger = logging.getLogger(__name__)


class FallbackPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = False

    def __init__(self,
                 featurizer=None,
                 max_history=None,
                 nlu_threshold=0.3,
                 core_threshold=0.3,
                 fallback_action_name="action_listen"):
        # type: (Optional[Featurizer]) -> None
        """Policy which executes a fallback action if NLU confidence is low
        or no other policy has a high-confidence prediction.

        :param nlu_threshold: minimum threshold for NLU confidence.
                              If intent prediction confidence is lower than this,
                              predict fallback action with confidence 1.0
        :param core_threshold: if NLU confidence threshold is met,
                               predict fallback action with confidence core_threshold.
                               If this is the highest confidence in the ensemble,
                               the fallback action will be executed.
        :fallback_action_name: name of the action to execute as a fallback.
        """

        self.featurizer = featurizer
        self.max_history = max_history
        self.nlu_threshold = nlu_threshold
        self.core_threshold = core_threshold
        self.fallback_action_name = fallback_action_name

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        result = [0.0] * domain.num_actions
        idx = domain.index_for_action(self.fallback_action_name)
        nlu_data = tracker.latest_message.parse_data
        nlu_confidence = nlu_data["intent"].get("confidence", 1.0)

        if nlu_confidence < self.nlu_threshold:
            score = 1.0
        else:
            score = self.core_threshold
        result[idx] = score

        return result

    def train(self, training_data, domain, **kwargs):
        # type: (DialogueTrainingData, Domain, **Any) -> None
        """Does nothing. This policy is deterministic."""

        pass

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""
        config_file = os.path.join(path, 'fallback_policy.json')
        meta = {
            "nlu_threshold": self.nlu_threshold,
            "core_threshold": self.core_threshold,
            "fallback_action_name": self.fallback_action_name
        }
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path, featurizer, max_history):
        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "fallback_policy.json")
            if os.path.isfile(meta_path):
                with io.open(meta_path) as f:
                    meta = json.loads(f.read())

        return cls(featurizer=featurizer,
                   max_history=max_history,
                   **meta)
