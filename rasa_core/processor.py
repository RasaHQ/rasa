from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
from types import LambdaType

from apscheduler.schedulers.background import BackgroundScheduler
from typing import Optional, List, Dict, Any
from typing import Text

from rasa_core.actions import Action
from rasa_core.actions.action import ActionRestart, ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage, InputChannel
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.dispatcher import Dispatcher
from rasa_core.domain import Domain
from rasa_core.events import Restarted, ReminderScheduled, Event
from rasa_core.events import UserUttered, ActionExecuted
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.ensemble import PolicyEnsemble
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker

scheduler = BackgroundScheduler()
scheduler.start()

logger = logging.getLogger(__name__)


class MessageProcessor(object):
    def __init__(self,
                 interpreter,  # type: NaturalLanguageInterpreter
                 policy_ensemble,  # type: PolicyEnsemble
                 domain,  # type: Domain
                 tracker_store,  # type: TrackerStore
                 max_number_of_predictions=10,  # type: int
                 message_preprocessor=None,  # type: Optional[LambdaType]
                 on_circuit_break=None  # type: Optional[LambdaType]
                 ):
        self.interpreter = interpreter
        self.policy_ensemble = policy_ensemble
        self.domain = domain
        self.tracker_store = tracker_store
        self.max_number_of_predictions = max_number_of_predictions
        self.message_preprocessor = message_preprocessor
        self.on_circuit_break = on_circuit_break

    def handle_channel(self, input_channel=None):
        # type: (InputChannel) -> None
        """Handles the input channel synchronously.

        Each message gets processed directly after it got received."""
        input_channel.start_sync_listening(self.handle_message)

    def handle_channel_asynchronous(self, message_queue):
        """Handles incoming messages from the message queue.

        An input channel should add messages to the queue asynchronously."""
        while True:
            message = message_queue.dequeue()
            if message is None:
                continue
            self.handle_message(message)

    def handle_message(self, message):
        # type: (UserMessage) -> Optional[List[Text]]
        """Handle a single message with this processor."""

        # preprocess message if necessary
        if self.message_preprocessor is not None:
            message.text = self.message_preprocessor(message.text)
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(message.sender_id)
        self._handle_message_with_tracker(message, tracker)
        self._predict_and_execute_next_action(message, tracker)
        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return [outgoing_message
                    for _, outgoing_message in message.output_channel.messages]
        else:
            return None

    def start_message_handling(self, message):
        # type: (UserMessage) -> Dict[Text, Any]

        # pre-process message if necessary
        if self.message_preprocessor is not None:
            message.text = self.message_preprocessor(message.text)

        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = self._get_tracker(message.sender_id)
        self._handle_message_with_tracker(message, tracker)

        # Log currently set slots
        self._log_slots(tracker)

        # action loop. predicts actions until we hit action listen
        if self._should_handle_message(tracker):
            return self._predict_next_and_return_state(tracker)
        else:
            return {"next_action": None,
                    "info": "Bot is currently paused and no restart was "
                            "received yet.",
                    "tracker": tracker.current_state()}

    def continue_message_handling(self, sender_id, executed_action, events):
        # type: (Text, Text, List[Event]) -> Dict[Text, Any]

        tracker = self._get_tracker(sender_id)
        if executed_action != ACTION_LISTEN_NAME:
            self._log_action_on_tracker(tracker, executed_action, events)
        if self._should_predict_another_action(executed_action, events):
            return self._predict_next_and_return_state(tracker)
        else:
            self._save_tracker(tracker)
            return {"next_action": None,
                    "info": "You do not need to call continue after action "
                            "listen got returned for the previous continue "
                            "call. You are expected to call 'parse' with the "
                            "next user message.",
                    "tracker": tracker.current_state()}

    def _predict_next_and_return_state(self, tracker):
        action = self._get_next_action(tracker)
        # save tracker state to continue conversation from this state
        if action.name() == ACTION_LISTEN_NAME:
            # action listen always get logged automatically - no need to
            # call continue
            self._log_action_on_tracker(tracker, action.name(), [])
        self._save_tracker(tracker)
        return {"next_action": action.name(),
                "tracker": tracker.current_state()}

    def _log_slots(self, tracker):
        # Log currently set slots
        slot_values = "\n".join(["\t{}: {}".format(s.name, s.value)
                                 for s in tracker.slots.values()])
        logger.debug("Current slot values: \n{}".format(slot_values))

    def handle_reminder(self, reminder_event, dispatcher):
        # type: (ReminderScheduled, Dispatcher) -> None
        """Handle a reminder that is triggered asynchronously."""

        def has_message_after_reminder(tracker):
            """If the user sent a message after the reminder got scheduled -
            it might be better to cancel it."""

            for e in reversed(tracker.events):
                if isinstance(e,
                              ReminderScheduled) and e.name == \
                        reminder_event.name:
                    return False
                elif isinstance(e, UserUttered):
                    return True
            return True  # tracker has probably been restarted

        tracker = self._get_tracker(dispatcher.sender_id)

        if (reminder_event.kill_on_user_message and
                has_message_after_reminder(tracker)):
            logger.debug("Canceled reminder because it is outdated. "
                         "(event: {} id: {})".format(reminder_event.action_name,
                                                     reminder_event.name))
        else:
            # necessary for proper featurization, otherwise the previous
            # unrelated message would influence featurization
            tracker.update(UserUttered.empty())
            action = self.domain.action_for_name(reminder_event.action_name)
            should_continue = self._run_action(action, tracker, dispatcher)
            if should_continue:
                user_msg = UserMessage(None,
                                       dispatcher.output_channel,
                                       dispatcher.sender_id)
                self._predict_and_execute_next_action(user_msg, tracker)
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)

    def _parse_message(self, message):
        # for testing - you can short-cut the NLU part with a message
        # in the format _intent[entity1=val1,entity=val2]
        # parse_data is a dict of intent & entities
        if message.text.startswith('_'):
            parse_data = RegexInterpreter().parse(message.text)
        else:
            parse_data = self.interpreter.parse(message.text)

        logger.debug("Received user message '{}' with intent '{}' "
                     "and entities  '{}'".format(message.text,
                                                 parse_data["intent"],
                                                 parse_data["entities"]))
        return parse_data

    def _handle_message_with_tracker(self, message, tracker):
        # type: (UserMessage, DialogueStateTracker) -> None

        parse_data = self._parse_message(message)

        # don't ever directly mutate the tracker - instead pass it events to log
        tracker.update(UserUttered(message.text, parse_data["intent"],
                                   parse_data["entities"], parse_data))
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
                                self.domain)
        # keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True
        num_predicted_actions = 0

        self._log_slots(tracker)

        # action loop. predicts actions until we hit action listen
        while should_predict_another_action and \
                self._should_handle_message(tracker) and \
                num_predicted_actions < self.max_number_of_predictions:
            # this actually just calls the policy's method by the same name
            action = self._get_next_action(tracker)

            should_predict_another_action = self._run_action(action,
                                                             tracker,
                                                             dispatcher)
            num_predicted_actions += 1

        if num_predicted_actions == self.max_number_of_predictions and \
                should_predict_another_action:
            # circuit breaker was tripped
            logger.warn(
                    "Circuit breaker tripped. Stopped predicting "
                    "more actions for sender '{}'".format(tracker.sender_id))
            if self.on_circuit_break:
                # call a registered callback
                self.on_circuit_break(tracker, dispatcher)

        logger.debug("Current topic: {}".format(tracker.topic.name))

    def _should_predict_another_action(self, action_name, events):
        is_listen_action = action_name == ACTION_LISTEN_NAME
        contains_restart = events and isinstance(events[0], Restarted)
        return not is_listen_action and not contains_restart

    def _schedule_reminders(self, events, dispatcher):
        # type: (List[Event], Dispatcher) -> None
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

    def _run_action(self, action, tracker, dispatcher):
        # events and return values are used to update
        # the tracker state after an action has been taken
        try:
            events = action.run(dispatcher, tracker, self.domain)
        except Exception as e:
            logger.error("Encountered an exception while running action '{}'. "
                         "Bot will continue, but the actions events are lost. "
                         "Make sure to fix the exception in your custom "
                         "code.".format(action.name()), )
            logger.error(e, exc_info=True)
            events = []
        self._log_action_on_tracker(tracker, action.name(), events)
        self._schedule_reminders(events, dispatcher)

        return self._should_predict_another_action(action.name(), events)

    def _log_action_on_tracker(self, tracker, action_name, events):
        # Ensures that the code still works even if a lazy programmer missed
        # to type `return []` at the end of an action or the run method
        # returns `None` for some other reason.
        if events is None:
            events = []

        logger.debug("Action '{}' ended with events '{}'".format(
                action_name, ['{}'.format(e) for e in events]))

        if action_name is not None:
            # log the action and its produced events
            tracker.update(ActionExecuted(action_name))

        for e in events:
            tracker.update(e)

    def _get_tracker(self, sender_id):
        # type: (Text) -> DialogueStateTracker

        sender_id = sender_id or UserMessage.DEFAULT_SENDER_ID
        tracker = self.tracker_store.get_or_create_tracker(sender_id)
        return tracker

    def _save_tracker(self, tracker):
        self.tracker_store.save(tracker)

    def _get_next_action(self, tracker):
        # type: (DialogueStateTracker) -> Action

        follow_up_action = tracker.follow_up_action
        if follow_up_action:
            tracker.clear_follow_up_action()
            if self.domain.index_for_action(
                    follow_up_action.name()) is not None:
                return follow_up_action
            else:
                logger.error(
                        "Trying to run unknown follow up action '{}'!"
                        "Instead of running that, we will ignore the action "
                        "and predict the next action.".format(follow_up_action))

        if tracker.latest_message.intent.get("name") == \
                self.domain.restart_intent:
            return ActionRestart()

        idx = self.policy_ensemble.predict_next_action(tracker, self.domain)
        return self.domain.action_for_index(idx)
