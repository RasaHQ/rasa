import logging
from typing import List, Text
import os
import json

from rasa_core.events import UserUtteranceReverted, ActionReverted, \
    UserUttered, ActionExecuted
from rasa_core.constants import FALLBACK_SCORE
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.policy import confidence_scores_for
from rasa_core.trackers import DialogueStateTracker
from rasa_core.domain import Domain
from rasa_core import utils

logger = logging.getLogger(__name__)


class TwoStageFallbackPolicy(FallbackPolicy):
    """ This policy handles low NLU confidence in multiple stages.

        If a NLU prediction has a low confidence store,
        the user is asked to confirm whether they really had this intent.
        If they confirm, the story continues as if the intent was recognized
        with high confidence from the beginning.
        If they deny, the user is asked to restate his intent.
        If the recognition for this clarification was confident the story
        continues as if the user had this intent from the beginning.
        If the clarification was not recognised with high confidence, the user
        is asked to confirm the recognized intent.
        If the user confirms the intent, the story continues as if the user had
        this intent from the beginning.
        If the user denies, an ultimate fallback action is triggered
        (e.g. a handoff to a human).
    """

    def __init__(self,
                 nlu_threshold: float = 0.3,
                 core_threshold: float = 0.3,
                 confirmation_action_name: Text = "action_ask_confirmation",
                 clarification_action_name: Text = "action_ask_clarification",
                 fallback_action_name: Text = "action_default_fallback",
                 confirm_intent_name: Text = "confirm",
                 deny_intent_name: Text = "deny",
                 ) -> None:
        """Create a new Two Stage Fallback policy.

                Args:
                    nlu_threshold: minimum threshold for NLU confidence.
                        If intent prediction confidence is lower than this,
                        predict fallback action with confidence 1.0.
                    core_threshold: if NLU confidence threshold is met,
                        predict fallback action with confidence `core_threshold`.
                        If this is the highest confidence in the ensemble,
                        the fallback action will be executed.
                    confirmation_action_name: This action is executed if the
                        user has to confirm their intent.
                    clarification_action_name: This action is executed if the
                        user should clarify / restate their intent.
                    fallback_action_name: This action is executed if the user
                        denies the recognised intent for the second time.
                    confirm_intent_name: If NLU recognises this intent, the
                        user has agreed to the suggested intent.
                    deny_intent_name: If NLU recognises this intent, the user
                        has denied the suggested intent.
                """
        super(TwoStageFallbackPolicy, self).__init__()

        self.nlu_threshold = nlu_threshold
        self.core_threshold = core_threshold
        self.confirmation_action_name = confirmation_action_name
        self.clarification_action_name = clarification_action_name
        self.fallback_action_name = fallback_action_name
        self.confirm_intent_name = confirm_intent_name
        self.deny_intent_name = deny_intent_name

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predicts the next action if NLU confidence is low.
        """

        nlu_data = tracker.latest_message.parse_data
        nlu_confidence = nlu_data["intent"].get("confidence", 1.0)
        last_intent_name = nlu_data['intent'].get('name', None)
        should_fallback = self.should_fallback(nlu_confidence,
                                               tracker.latest_action_name)

        if self.__is_user_input_expected(tracker):
            result = confidence_scores_for('action_listen', FALLBACK_SCORE,
                                           domain)
        elif self.__user_confirmed(last_intent_name, tracker):
            logger.debug("User confirmed suggested intent.")
            result = self.__results_for_user_confirmed(tracker, domain)
        elif self.__user_denied(last_intent_name, tracker):
            logger.debug("User denied suggested intent.")
            result = self.__results_for_user_denied(tracker, domain)
        elif self.__user_clarified(tracker):
            logger.debug("User clarified intent.")
            result = self.__results_for_clarification(tracker, domain,
                                                      should_fallback)
        elif tracker.last_executed_has(name=self.confirmation_action_name):
            logger.debug("User tried to clarify instead of confirming.")
            result = self.__results_for_early_clarification(should_fallback,
                                                                tracker, domain)
        elif should_fallback:
            logger.debug("User has to confirm intent.")
            result = confidence_scores_for(self.confirmation_action_name,
                                           FALLBACK_SCORE,
                                           domain)
        else:
            result = self.fallback_scores(domain, self.core_threshold)

        return result

    def __is_user_input_expected(self, tracker: DialogueStateTracker) -> bool:
        return tracker.latest_action_name in [self.confirmation_action_name,
                                              self.clarification_action_name,
                                              ]

    def __user_confirmed(self, last_intent: Text,
                         tracker: DialogueStateTracker) -> bool:
        return tracker.last_executed_has(name=self.confirmation_action_name) \
               and last_intent == self.confirm_intent_name

    def __user_denied(self, last_intent: Text,
                      tracker: DialogueStateTracker) -> bool:
        return tracker.last_executed_has(name=self.confirmation_action_name) \
               and last_intent == self.deny_intent_name

    def __results_for_user_confirmed(self, tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        _revert_confirmation_actions(tracker)

        intent = tracker.get_last_event_for(UserUttered, skip=1)
        intent.parse_data['intent']['confidence'] = FALLBACK_SCORE

        clarification = tracker.last_executed_has(
            self.clarification_action_name,
            skip=1)
        if clarification:
            _revert_clarification_actions(intent, tracker)

        return self.fallback_scores(domain, self.core_threshold)

    def __results_for_user_denied(self, tracker: DialogueStateTracker,
                                  domain: Domain) -> List[float]:
        has_denied_before = tracker.last_executed_has(
            self.clarification_action_name,
            skip=1)

        if has_denied_before:
            # TODO: Revert old events here?
            return confidence_scores_for(self.fallback_action_name,
                                         FALLBACK_SCORE, domain)
        else:
            return confidence_scores_for(self.clarification_action_name,
                                         FALLBACK_SCORE, domain)

    def __user_clarified(self, tracker: DialogueStateTracker) -> bool:
        return tracker.last_executed_has(name=self.clarification_action_name)

    def __results_for_clarification(self, tracker: DialogueStateTracker,
                                    domain: Domain,
                                    should_fallback: bool) -> List[float]:

        if should_fallback:
            logger.debug("Clarification of intent was not clear enough.")
            return confidence_scores_for(self.confirmation_action_name,
                                         FALLBACK_SCORE,
                                         domain)
        else:
            logger.debug('User clarified intent successfully.')
            last_intent = tracker.get_last_event_for(UserUttered)
            _revert_clarification_actions(last_intent, tracker)

            return self.fallback_scores(domain, self.core_threshold)

    def __results_for_early_clarification(self, should_fallback: bool,
                                          tracker: DialogueStateTracker,
                                          domain: Domain) -> List[float]:
        # keep clarification intent separate
        clarification = tracker.get_last_event_for(UserUttered)
        # remove utterances and confirmation requests
        tracker.update(UserUtteranceReverted())
        _revert_confirmation_actions(tracker)
        # re-add clarification
        tracker.update(clarification)

        if should_fallback:
            return confidence_scores_for(self.fallback_action_name,
                                         FALLBACK_SCORE, domain)
        else:
            return self.fallback_scores(domain, self.core_threshold)

    def persist(self, path: Text) -> None:
        """Persists the policy to storage."""
        config_file = os.path.join(path, 'two_stage_fallback_policy.json')
        meta = {
            "nlu_threshold": self.nlu_threshold,
            "core_threshold": self.core_threshold,
            "confirmation_action_name": self.confirmation_action_name,
            "clarification_action_name": self.clarification_action_name,
            "fallback_action_name": self.fallback_action_name,
            "confirm_intent_name": self.confirm_intent_name,
            "deny_intent_name": self.deny_intent_name,
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


def _revert_clarification_actions(last_intent: UserUttered,
                                   tracker: DialogueStateTracker) \
    -> None:
    from rasa_core.actions.action import ACTION_LISTEN_NAME
    revert_events = [UserUtteranceReverted(),  # remove clarification
                     ActionReverted(),  # remove clarification request
                     UserUtteranceReverted(),  # remove feedback
                     ActionReverted(),  # remove confirmation request
                     UserUtteranceReverted(),  # remove false intent
                     ActionReverted(),
                     # replace action with action listen
                     ActionExecuted(action_name=ACTION_LISTEN_NAME),
                     last_intent,  # add right intent
                     ]
    [tracker.update(e) for e in revert_events]


def _revert_confirmation_actions(tracker):
    # remove user confirmation
    tracker.update(UserUtteranceReverted())
    # remove actions
    tracker.update(ActionReverted())
