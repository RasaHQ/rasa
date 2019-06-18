import json
import logging
import os
from typing import List, Text

import rasa.utils.io
from rasa.core import utils
from rasa.core.actions.action import (
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_LISTEN_NAME,
)
from rasa.core.constants import USER_INTENT_OUT_OF_SCOPE
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.policy import confidence_scores_for
from rasa.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


def has_user_rephrased(tracker: DialogueStateTracker) -> bool:
    return tracker.last_executed_action_has(ACTION_DEFAULT_ASK_REPHRASE_NAME)


class TwoStageFallbackPolicy(FallbackPolicy):
    """ This policy handles low NLU confidence in multiple stages.

        If a NLU prediction has a low confidence score,
        the user is asked to affirm whether they really had this intent.
        If they affirm, the story continues as if the intent was classified
        with high confidence from the beginning.
        If they deny, the user is asked to rephrase his intent.
        If the classification for the rephrased intent was confident, the story
        continues as if the user had this intent from the beginning.
        If the rephrased intent was not classified with high confidence,
        the user is asked to affirm the classified intent.
        If the user affirm the intent, the story continues as if the user had
        this intent from the beginning.
        If the user denies, an ultimate fallback action is triggered
        (e.g. a hand-off to a human).
    """

    def __init__(
        self,
        priority: int = 4,
        nlu_threshold: float = 0.3,
        core_threshold: float = 0.3,
        fallback_core_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        fallback_nlu_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
        deny_suggestion_intent_name: Text = USER_INTENT_OUT_OF_SCOPE,
    ) -> None:
        """Create a new Two-stage Fallback policy.

        Args:
            nlu_threshold: minimum threshold for NLU confidence.
                If intent prediction confidence is lower than this,
                predict fallback action with confidence 1.0.
            core_threshold: if NLU confidence threshold is met,
                predict fallback action with confidence
                `core_threshold`. If this is the highest confidence in
                the ensemble, the fallback action will be executed.
            fallback_core_action_name: This action is executed if the Core
                threshold is not met.
            fallback_nlu_action_name: This action is executed if the user
                denies the recognised intent for the second time.
            deny_suggestion_intent_name: The name of the intent which is used
                 to detect that the user denies the suggested intents.
        """
        super(TwoStageFallbackPolicy, self).__init__(
            priority, nlu_threshold, core_threshold, fallback_core_action_name
        )

        self.fallback_nlu_action_name = fallback_nlu_action_name
        self.deny_suggestion_intent_name = deny_suggestion_intent_name

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predicts the next action if NLU confidence is low.
        """

        if self.deny_suggestion_intent_name not in domain.intents:
            raise InvalidDomain(
                "The intent {} must be present in the "
                "domain file to use the "
                "`TwoStageFallbackPolicy`."
                "".format(self.deny_suggestion_intent_name)
            )

        nlu_data = tracker.latest_message.parse_data
        nlu_confidence = nlu_data["intent"].get("confidence", 1.0)
        last_intent_name = nlu_data["intent"].get("name", None)
        should_nlu_fallback = self.should_nlu_fallback(
            nlu_confidence, tracker.latest_action_name
        )
        user_rephrased = has_user_rephrased(tracker)

        if self._is_user_input_expected(tracker):
            result = confidence_scores_for(ACTION_LISTEN_NAME, 1.0, domain)
        elif self._has_user_denied(last_intent_name, tracker):
            logger.debug(
                "User '{}' denied suggested intents.".format(tracker.sender_id)
            )
            result = self._results_for_user_denied(tracker, domain)
        elif user_rephrased and should_nlu_fallback:
            logger.debug(
                "Ambiguous rephrasing of user '{}' "
                "for intent '{}'".format(tracker.sender_id, last_intent_name)
            )
            result = confidence_scores_for(
                ACTION_DEFAULT_ASK_AFFIRMATION_NAME, 1.0, domain
            )
        elif user_rephrased:
            logger.debug("User '{}' rephrased intent".format(tracker.sender_id))
            result = confidence_scores_for(
                ACTION_REVERT_FALLBACK_EVENTS_NAME, 1.0, domain
            )
        elif tracker.last_executed_action_has(ACTION_DEFAULT_ASK_AFFIRMATION_NAME):
            if not should_nlu_fallback:
                logger.debug(
                    "User '{}' affirmed intent '{}'"
                    "".format(tracker.sender_id, last_intent_name)
                )
                result = confidence_scores_for(
                    ACTION_REVERT_FALLBACK_EVENTS_NAME, 1.0, domain
                )
            else:
                result = confidence_scores_for(
                    self.fallback_nlu_action_name, 1.0, domain
                )
        elif should_nlu_fallback:
            logger.debug(
                "User '{}' has to affirm intent '{}'.".format(
                    tracker.sender_id, last_intent_name
                )
            )
            result = confidence_scores_for(
                ACTION_DEFAULT_ASK_AFFIRMATION_NAME, 1.0, domain
            )
        else:
            logger.debug(
                "NLU confidence threshold met, confidence of "
                "fallback action set to core threshold ({}).".format(
                    self.core_threshold
                )
            )
            result = self.fallback_scores(domain, self.core_threshold)

        return result

    def _is_user_input_expected(self, tracker: DialogueStateTracker) -> bool:
        return tracker.latest_action_name in [
            ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
            ACTION_DEFAULT_ASK_REPHRASE_NAME,
            self.fallback_action_name,
        ]

    def _has_user_denied(
        self, last_intent: Text, tracker: DialogueStateTracker
    ) -> bool:
        return (
            tracker.last_executed_action_has(ACTION_DEFAULT_ASK_AFFIRMATION_NAME)
            and last_intent == self.deny_suggestion_intent_name
        )

    def _results_for_user_denied(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        has_denied_before = tracker.last_executed_action_has(
            ACTION_DEFAULT_ASK_REPHRASE_NAME, skip=1
        )

        if has_denied_before:
            return confidence_scores_for(self.fallback_nlu_action_name, 1.0, domain)
        else:
            return confidence_scores_for(ACTION_DEFAULT_ASK_REPHRASE_NAME, 1.0, domain)

    def persist(self, path: Text) -> None:
        """Persists the policy to storage."""
        config_file = os.path.join(path, "two_stage_fallback_policy.json")
        meta = {
            "priority": self.priority,
            "nlu_threshold": self.nlu_threshold,
            "core_threshold": self.core_threshold,
            "fallback_core_action_name": self.fallback_action_name,
            "fallback_nlu_action_name": self.fallback_nlu_action_name,
            "deny_suggestion_intent_name": self.deny_suggestion_intent_name,
        }
        rasa.utils.io.create_directory_for_file(config_file)
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path: Text) -> "FallbackPolicy":
        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "two_stage_fallback_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(rasa.utils.io.read_file(meta_path))

        return cls(**meta)
