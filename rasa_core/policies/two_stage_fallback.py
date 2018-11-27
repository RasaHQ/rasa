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

    def __init__(self,
                 nlu_threshold: float = 0.3,
                 core_threshold: float = 0.3,
                 confirmation_action_name: Text = "action_ask_confirmation",
                 clarification_action_name: Text = "action_ask_clarification",
                 fallback_action_name: Text = "action_default_fallback",
                 confirm_intent_name: Text = "confirm",
                 deny_intent_name: Text = "deny",
                 ) -> None:

        super(TwoStageFallbackPolicy, self).__init__()

        self.nlu_threshold = nlu_threshold
        self.core_threshold = core_threshold
        self.confirmation_action_name = confirmation_action_name
        self.clarification_action_name = clarification_action_name
        self.fallback_action_name = fallback_action_name
        self.confirm_intent_name = confirm_intent_name
        self.deny_intent_name = deny_intent_name

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

    def __revert_clarification_actions(self, last_intent: UserUttered,
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

    def __results_for_user_confirmed(self, tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        # remove user confirmation
        tracker.update(UserUtteranceReverted())
        # remove actions
        tracker.update(ActionReverted())

        intent = tracker.get_last_event_for(UserUttered, skip=1)
        intent.parse_data['intent']['confidence'] = FALLBACK_SCORE

        clarification = tracker.last_executed_has(
            self.clarification_action_name,
            skip=1)
        if clarification:
            self.__revert_clarification_actions(intent, tracker)

        return self.fallback_scores(domain, self.core_threshold)

    def __results_for_user_denied(self, tracker: DialogueStateTracker,
                                  domain: Domain) -> List[float]:
        has_denied_before = tracker.last_executed_has(
            self.clarification_action_name,
            skip=1)

        if has_denied_before:
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
            self.__revert_clarification_actions(last_intent, tracker)

            return self.fallback_scores(domain, self.core_threshold)

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predicts a fallback action if NLU confidence is low
            or no other policy has a high-confidence prediction"""

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
        elif should_fallback:
            logger.debug("User has to confirm intent.")
            result = confidence_scores_for(self.confirmation_action_name,
                                           FALLBACK_SCORE,
                                           domain)
        else:
            result = self.fallback_scores(domain, self.core_threshold)

        return result

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
