import time

import json
import logging
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import UnknownTimeZoneError
from types import LambdaType
from typing import Optional, List, Dict, Any, Tuple
from typing import Text

from rasa_core.actions import Action
from rasa_core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ActionExecutionRejection)
from rasa_core.channels import CollectingOutputChannel
from rasa_core.channels import UserMessage
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import Domain
from rasa_core.events import ReminderScheduled, Event
from rasa_core.events import SlotSet
from rasa_core.events import (
    UserUttered,
    ActionExecuted,
    BotUttered,
    ActionExecutionRejected)
from rasa_core.interpreter import (
    NaturalLanguageInterpreter,
    RasaNLUHttpInterpreter,
    INTENT_MESSAGE_PREFIX)
from rasa_core.interpreter import RegexInterpreter
from rasa_core.nlg import NaturalLanguageGenerator
from rasa_core.policies.ensemble import PolicyEnsemble
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker, EventVerbosity
from rasa_core.utils import EndpointConfig

logger = logging.getLogger(__name__)

try:
    scheduler = BackgroundScheduler()
    scheduler.start()
except UnknownTimeZoneError:
    logger.warning("apscheduler failed to start. "
                   "This is probably because your system timezone is not set"
                   "Set it with e.g. echo \"Europe/Berlin\" > /etc/timezone")


class MessageProcessor(object):
    def __init__(self,
                 interpreter: NaturalLanguageInterpreter,
                 policy_ensemble: PolicyEnsemble,
                 domain: Domain,
                 tracker_store: TrackerStore,
                 generator: NaturalLanguageGenerator,
                 action_endpoint: Optional[EndpointConfig] = None,
                 max_number_of_predictions: int = 10,
                 message_preprocessor: Optional[LambdaType] = None,
                 on_circuit_break: Optional[LambdaType] = None
                 ):
        self.interpreter = interpreter
        self.nlg = generator
        self.policy_ensemble = policy_ensemble
        self.domain = domain
        self.tracker_store = tracker_store
        self.max_number_of_predictions = max_number_of_predictions
        self.message_preprocessor = message_preprocessor
        self.on_circuit_break = on_circuit_break
        self.action_endpoint = action_endpoint

    def handle_message(self, message: UserMessage) -> Optional[List[Text]]:
        """Handle a single message with this processor."""

        # preprocess message if necessary
        tracker = self.log_message(message)
        if not tracker:
            return None

        self._predict_and_execute_next_action(message, tracker)
        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return message.output_channel.messages
        else:
            return None

    def predict_next(self, sender_id: Text) -> Optional[Dict[Text, Any]]:

        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(sender_id)
        if not tracker:
            logger.warning("Failed to retrieve or create tracker for sender "
                           "'{}'.".format(sender_id))
            return None

        probabilities, policy = \
            self._get_next_action_probabilities(tracker)
        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)
        scores = [{"action": a, "score": p}
                  for a, p in zip(self.domain.action_names, probabilities)]
        return {
            "scores": scores,
            "policy": policy,
            "confidence": np.max(probabilities),
            "tracker": tracker.current_state(EventVerbosity.AFTER_RESTART)
        }

    def log_message(self,
                    message: UserMessage) -> Optional[DialogueStateTracker]:

        # preprocess message if necessary
        if self.message_preprocessor is not None:
            message.text = self.message_preprocessor(message.text)
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(message.sender_id)
        if tracker:
            self._handle_message_with_tracker(message, tracker)
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)
        else:
            logger.warning("Failed to retrieve or create tracker for sender "
                           "'{}'.".format(message.sender_id))
        return tracker

    def execute_action(self,
                       sender_id: Text,
                       action_name: Text,
                       dispatcher: Dispatcher,
                       policy: Text,
                       confidence: float
                       ) -> Optional[DialogueStateTracker]:

        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(sender_id)
        if tracker:
            action = self._get_action(action_name)
            self._run_action(action, tracker, dispatcher, policy,
                             confidence)

            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)
        else:
            logger.warning("Failed to retrieve or create tracker for sender "
                           "'{}'.".format(sender_id))
        return tracker

    def predict_next_action(self,
                            tracker: DialogueStateTracker
                            ) -> Tuple[Action, Text, float]:
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use
        ML to predict the action. Returns the index of the next action."""

        probabilities, policy = self._get_next_action_probabilities(tracker)

        max_index = int(np.argmax(probabilities))
        action = self.domain.action_for_index(max_index, self.action_endpoint)
        logger.debug("Predicted next action '{}' with prob {:.2f}.".format(
            action.name(), probabilities[max_index]))
        return action, policy, probabilities[max_index]

    @staticmethod
    def _is_reminder_still_valid(tracker: DialogueStateTracker,
                                 reminder_event: ReminderScheduled
                                 ) -> bool:
        """Check if the conversation has been restarted after reminder."""

        for e in reversed(tracker.applied_events()):
            if (isinstance(e, ReminderScheduled) and
                    e.name == reminder_event.name):
                return True
        return False  # not found in applied events --> has been restarted

    @staticmethod
    def _has_message_after_reminder(tracker: DialogueStateTracker,
                                    reminder_event: ReminderScheduled
                                    ) -> bool:
        """Check if the user sent a message after the reminder."""

        for e in reversed(tracker.events):
            if (isinstance(e, ReminderScheduled) and
                    e.name == reminder_event.name):
                return False
            elif isinstance(e, UserUttered) and e.text:
                return True
        return True  # tracker has probably been restarted

    def handle_reminder(self,
                        reminder_event: ReminderScheduled,
                        dispatcher: Dispatcher
                        ) -> None:
        """Handle a reminder that is triggered asynchronously."""

        tracker = self._get_tracker(dispatcher.sender_id)

        if not tracker:
            logger.warning("Failed to retrieve or create tracker for sender "
                           "'{}'.".format(dispatcher.sender_id))
            return None

        if (reminder_event.kill_on_user_message and
                self._has_message_after_reminder(tracker, reminder_event) or not
                self._is_reminder_still_valid(tracker, reminder_event)):
            logger.debug("Canceled reminder because it is outdated. "
                         "(event: {} id: {})".format(reminder_event.action_name,
                                                     reminder_event.name))
        else:
            # necessary for proper featurization, otherwise the previous
            # unrelated message would influence featurization
            tracker.update(UserUttered.empty())
            action = self._get_action(reminder_event.action_name)
            should_continue = self._run_action(action, tracker, dispatcher)
            if should_continue:
                user_msg = UserMessage(None,
                                       dispatcher.output_channel,
                                       dispatcher.sender_id)
                self._predict_and_execute_next_action(user_msg, tracker)
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)

    @staticmethod
    def _log_slots(tracker):
        # Log currently set slots
        slot_values = "\n".join(["\t{}: {}".format(s.name, s.value)
                                 for s in tracker.slots.values()])
        logger.debug("Current slot values: \n{}".format(slot_values))

    def _get_action(self, action_name):
        return self.domain.action_for_name(action_name, self.action_endpoint)

    def _parse_message(self, message):
        # for testing - you can short-cut the NLU part with a message
        # in the format /intent{"entity1": val1, "entity2": val2}
        # parse_data is a dict of intent & entities
        if message.text.startswith(INTENT_MESSAGE_PREFIX):
            parse_data = RegexInterpreter().parse(message.text)
        elif isinstance(self.interpreter, RasaNLUHttpInterpreter):
            parse_data = self.interpreter.parse(message.text,
                                                message.message_id)
        else:
            parse_data = self.interpreter.parse(message.text)

        logger.debug("Received user message '{}' with intent '{}' "
                     "and entities '{}'".format(message.text,
                                                parse_data["intent"],
                                                parse_data["entities"]))
        return parse_data

    def _handle_message_with_tracker(self,
                                     message: UserMessage,
                                     tracker: DialogueStateTracker) -> None:

        if message.parse_data:
            parse_data = message.parse_data
        else:
            parse_data = self._parse_message(message)

        # don't ever directly mutate the tracker
        # - instead pass its events to log
        tracker.update(UserUttered(message.text, parse_data["intent"],
                                   parse_data["entities"], parse_data,
                                   input_channel=message.input_channel,
                                   message_id=message.message_id))
        # store all entities as slots
        for e in self.domain.slots_for_entities(parse_data["entities"]):
            tracker.update(e)

        logger.debug("Logged UserUtterance - "
                     "tracker now has {} events".format(len(tracker.events)))

    def _should_handle_message(self, tracker):
        return (not tracker.is_paused() or
                tracker.latest_message.intent.get("name") ==
                self.domain.restart_intent)

    def _predict_and_execute_next_action(self, message, tracker):
        # this will actually send the response to the user

        dispatcher = Dispatcher(message.sender_id,
                                message.output_channel,
                                self.nlg)
        # keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True
        num_predicted_actions = 0

        self._log_slots(tracker)

        # action loop. predicts actions until we hit action listen
        while (should_predict_another_action and
               self._should_handle_message(tracker) and
               num_predicted_actions < self.max_number_of_predictions):
            # this actually just calls the policy's method by the same name
            action, policy, confidence = self.predict_next_action(tracker)

            should_predict_another_action = self._run_action(action,
                                                             tracker,
                                                             dispatcher,
                                                             policy,
                                                             confidence)
            num_predicted_actions += 1

        if (num_predicted_actions == self.max_number_of_predictions and
                should_predict_another_action):
            # circuit breaker was tripped
            logger.warning(
                "Circuit breaker tripped. Stopped predicting "
                "more actions for sender '{}'".format(tracker.sender_id))
            if self.on_circuit_break:
                # call a registered callback
                self.on_circuit_break(tracker, dispatcher)

    # noinspection PyUnusedLocal
    @staticmethod
    def should_predict_another_action(action_name, events):
        is_listen_action = action_name == ACTION_LISTEN_NAME
        return not is_listen_action

    def _schedule_reminders(self, events: List[Event],
                            dispatcher: Dispatcher) -> None:
        """Uses the scheduler to time a job to trigger the passed reminder.

        Reminders with the same `id` property will overwrite one another
        (i.e. only one of them will eventually run)."""

        if events is not None:
            for e in events:
                if isinstance(e, ReminderScheduled):
                    scheduler.add_job(self.handle_reminder, "date",
                                      run_date=e.trigger_date_time,
                                      args=[e, dispatcher],
                                      id=e.name,
                                      replace_existing=True)

    def _run_action(self, action, tracker, dispatcher, policy=None,
                    confidence=None):
        # events and return values are used to update
        # the tracker state after an action has been taken
        try:
            events = action.run(dispatcher, tracker, self.domain)
        except ActionExecutionRejection:
            events = [ActionExecutionRejected(action.name(),
                                              policy, confidence)]
            tracker.update(events[0])
            return self.should_predict_another_action(action.name(), events)
        except Exception as e:
            logger.error("Encountered an exception while running action '{}'. "
                         "Bot will continue, but the actions events are lost. "
                         "Make sure to fix the exception in your custom "
                         "code.".format(action.name()))
            logger.debug(e, exc_info=True)
            events = []

        self._log_action_on_tracker(tracker, action.name(), events, policy,
                                    confidence)
        self.log_bot_utterances_on_tracker(tracker, dispatcher)
        self._schedule_reminders(events, dispatcher)

        return self.should_predict_another_action(action.name(), events)

    def _warn_about_new_slots(self, tracker, action_name, events):
        # these are the events from that action we have seen during training

        if action_name not in self.policy_ensemble.action_fingerprints:
            return

        fp = self.policy_ensemble.action_fingerprints[action_name]
        slots_seen_during_train = fp.get("slots", set())
        for e in events:
            if isinstance(e, SlotSet) and e.key not in slots_seen_during_train:
                s = tracker.slots.get(e.key)
                if s and s.has_features():
                    if e.key == 'requested_slot' and tracker.active_form:
                        pass
                    else:
                        logger.warning(
                            "Action '{0}' set a slot type '{1}' that "
                            "it never set during the training. This "
                            "can throw of the prediction. Make sure to "
                            "include training examples in your stories "
                            "for the different types of slots this "
                            "action can return. Remember: you need to "
                            "set the slots manually in the stories by "
                            "adding '- slot{{\"{1}\": {2}}}' "
                            "after the action."
                            "".format(action_name, e.key,
                                      json.dumps(e.value)))

    @staticmethod
    def log_bot_utterances_on_tracker(tracker: DialogueStateTracker,
                                      dispatcher: Dispatcher) -> None:

        if dispatcher.latest_bot_messages:
            for m in dispatcher.latest_bot_messages:
                bot_utterance = BotUttered(text=m.text, data=m.data)
                logger.debug("Bot utterance '{}'".format(bot_utterance))
                tracker.update(bot_utterance)

            dispatcher.latest_bot_messages = []

    def _log_action_on_tracker(self, tracker, action_name, events, policy,
                               confidence):
        # Ensures that the code still works even if a lazy programmer missed
        # to type `return []` at the end of an action or the run method
        # returns `None` for some other reason.
        if events is None:
            events = []

        logger.debug("Action '{}' ended with events '{}'".format(
            action_name, ['{}'.format(e) for e in events]))

        self._warn_about_new_slots(tracker, action_name, events)

        if action_name is not None:
            # log the action and its produced events
            tracker.update(ActionExecuted(action_name, policy, confidence))

        for e in events:
            # this makes sure the events are ordered by timestamp -
            # since the event objects are created somewhere else,
            # the timestamp would indicate a time before the time
            # of the action executed
            e.timestamp = time.time()
            tracker.update(e)

    def _get_tracker(self, sender_id: Text) -> Optional[DialogueStateTracker]:

        sender_id = sender_id or UserMessage.DEFAULT_SENDER_ID
        tracker = self.tracker_store.get_or_create_tracker(sender_id)
        return tracker

    def _save_tracker(self, tracker):
        self.tracker_store.save(tracker)

    def _prob_array_for_action(self,
                               action_name: Text
                               ) -> Tuple[Optional[List[float]], None]:
        idx = self.domain.index_for_action(action_name)
        if idx is not None:
            result = [0.0] * self.domain.num_actions
            result[idx] = 1.0
            return result, None
        else:
            return None, None

    def _get_next_action_probabilities(
        self,
        tracker: DialogueStateTracker
    ) -> Tuple[Optional[List[float]], Optional[Text]]:

        followup_action = tracker.followup_action
        if followup_action:
            tracker.clear_followup_action()
            result = self._prob_array_for_action(followup_action)
            if result:
                return result
            else:
                logger.error(
                    "Trying to run unknown follow up action '{}'!"
                    "Instead of running that, we will ignore the action "
                    "and predict the next action.".format(followup_action))

        if (tracker.latest_message.intent.get("name") ==
                self.domain.restart_intent):
            return self._prob_array_for_action(ACTION_RESTART_NAME)
        return self.policy_ensemble.probabilities_using_best_policy(
            tracker, self.domain)
