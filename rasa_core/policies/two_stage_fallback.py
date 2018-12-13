import json
import logging
import os
from typing import List, Text

from rasa_core import utils
from rasa_core.actions.action import (ACTION_REVERT_FALLBACK_EVENTS_NAME,
                                      ACTION_DEFAULT_FALLBACK_NAME,
                                      ACTION_DEFAULT_ASK_REPHRASE_NAME,
                                      ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
                                      ACTION_LISTEN_NAME)
from rasa_core.constants import (FALLBACK_SCORE, USER_INTENT_AFFIRM,
                                 USER_INTENT_DENY)
from rasa_core.domain import Domain, InvalidDomain
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.policy import confidence_scores_for
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


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
        (e.g. a handoff to a human).
    """

    def __init__(self,
                 nlu_threshold: float = 0.3,
                 core_threshold: float = 0.3,
                 fallback_action_name: Text = ACTION_DEFAULT_FALLBACK_NAME,
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
            fallback_action_name: This action is executed if the user
                denies the recognised intent for the second time.
        """
        super(TwoStageFallbackPolicy, self).__init__(
            nlu_threshold,
            core_threshold,
            fallback_action_name)

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predicts the next action if NLU confidence is low.
        """

        if (USER_INTENT_AFFIRM not in domain.intents or
                USER_INTENT_DENY not in domain.intents):
            raise InvalidDomain('The intents {} and {} must be present in the '
                                'domain file to use this policy.'.format(
                                    USER_INTENT_AFFIRM, USER_INTENT_AFFIRM))

        nlu_data = tracker.latest_message.parse_data
        nlu_confidence = nlu_data["intent"].get("confidence", 1.0)
        last_intent_name = nlu_data['intent'].get('name', None)
        should_fallback = self.should_fallback(nlu_confidence,
                                               tracker.latest_action_name)
        user_rephrased = has_user_rephrased(tracker)

        if self._is_user_input_expected(tracker):
            result = confidence_scores_for(ACTION_LISTEN_NAME, FALLBACK_SCORE,
                                           domain)
        elif _has_user_denied(last_intent_name, tracker):
            logger.debug("User '{}' denied suggested intents.".format(
                tracker.sender_id))
            result = self._results_for_user_denied(tracker, domain)
        elif user_rephrased and should_fallback:
            logger.debug("Ambiguous rephrasing of user '{}' "
                         "for intent '{}'".format(tracker.sender_id,
                                                  last_intent_name))
            result = confidence_scores_for(ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
                                           FALLBACK_SCORE,
                                           domain)
        elif has_user_affirmed(last_intent_name, tracker) or user_rephrased:
            logger.debug("User '{}' affirmed intent by affirmation or "
                         "rephrasing.".format(tracker.sender_id))
            result = confidence_scores_for(ACTION_REVERT_FALLBACK_EVENTS_NAME,
                                           FALLBACK_SCORE, domain)
        elif tracker.last_executed_action_has(
                ACTION_DEFAULT_ASK_AFFIRMATION_NAME):
            if not should_fallback:
                logger.debug("User '{}' rephrased intent '{}' instead "
                             "of affirming.".format(tracker.sender_id,
                                                    last_intent_name))
                result = confidence_scores_for(
                    ACTION_REVERT_FALLBACK_EVENTS_NAME,
                    FALLBACK_SCORE, domain)
            else:
                result = confidence_scores_for(self.fallback_action_name,
                                               FALLBACK_SCORE, domain)
        elif should_fallback:
            logger.debug("User '{}' has to affirm intent '{}'.".format(
                tracker.sender_id, last_intent_name))
            result = confidence_scores_for(ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
                                           FALLBACK_SCORE,
                                           domain)
        else:
            result = self.fallback_scores(domain, self.core_threshold)

        return result

    def _is_user_input_expected(self, tracker: DialogueStateTracker) -> bool:
        return tracker.latest_action_name in [
            ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
            ACTION_DEFAULT_ASK_REPHRASE_NAME,
            self.fallback_action_name]

    def _results_for_user_denied(self, tracker: DialogueStateTracker,
                                 domain: Domain) -> List[float]:
        has_denied_before = tracker.last_executed_action_has(
            ACTION_DEFAULT_ASK_REPHRASE_NAME,
            skip=1)

        if has_denied_before:
            return confidence_scores_for(self.fallback_action_name,
                                         FALLBACK_SCORE, domain)
        else:
            return confidence_scores_for(ACTION_DEFAULT_ASK_REPHRASE_NAME,
                                         FALLBACK_SCORE, domain)

    def persist(self, path: Text) -> None:
        """Persists the policy to storage."""
        config_file = os.path.join(path, 'two_stage_fallback_policy.json')
        meta = {
            "nlu_threshold": self.nlu_threshold,
            "core_threshold": self.core_threshold,
            "fallback_action_name": self.fallback_action_name,
        }
        utils.create_dir_for_file(config_file)
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path: Text) -> 'FallbackPolicy':
        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "two_stage_fallback_policy.json")
            if os.path.isfile(meta_path):
                meta = json.loads(utils.read_file(meta_path))

        return cls(**meta)


def has_user_affirmed(last_intent: Text,
                      tracker: DialogueStateTracker) -> bool:
    return (tracker.last_executed_action_has(
        ACTION_DEFAULT_ASK_AFFIRMATION_NAME) and
        last_intent == USER_INTENT_AFFIRM)


def _has_user_denied(last_intent: Text,
                     tracker: DialogueStateTracker) -> bool:
    return (tracker.last_executed_action_has(
        ACTION_DEFAULT_ASK_AFFIRMATION_NAME) and
        last_intent == USER_INTENT_DENY)


def has_user_rephrased(tracker: DialogueStateTracker) -> bool:
    return tracker.last_executed_action_has(
        ACTION_DEFAULT_ASK_REPHRASE_NAME)
