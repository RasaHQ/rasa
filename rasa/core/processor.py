import logging
import os
import time
from types import LambdaType
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import rasa.shared.utils.io
import rasa.core.actions.action
from rasa.core import jobs
from rasa.core.actions.action import Action
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
import rasa.core.utils
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.constants import (
    USER_INTENT_RESTART,
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    REQUESTED_SLOT,
    SLOTS,
    FOLLOWUP_ACTION,
    SESSION_START_METADATA_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecutionRejected,
    BotUttered,
    Event,
    ReminderCancelled,
    ReminderScheduled,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.slots import Slot
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    KEY_SLOT_NAME,
    KEY_ACTION,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.shared.constants import (
    INTENT_MESSAGE_PREFIX,
    DOCS_URL_DOMAINS,
    DEFAULT_SENDER_ID,
    DOCS_URL_POLICIES,
    UTTER_PREFIX,
    DOCS_URL_SLOTS,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.lock_store import LockStore
from rasa.core.policies.ensemble import PolicyEnsemble
import rasa.core.tracker_store
import rasa.shared.core.trackers
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

MAX_NUMBER_OF_PREDICTIONS = int(os.environ.get("MAX_NUMBER_OF_PREDICTIONS", "10"))


class MessageProcessor:
    def __init__(
        self,
        interpreter: NaturalLanguageInterpreter,
        policy_ensemble: PolicyEnsemble,
        domain: Domain,
        tracker_store: rasa.core.tracker_store.TrackerStore,
        lock_store: LockStore,
        generator: NaturalLanguageGenerator,
        action_endpoint: Optional[EndpointConfig] = None,
        max_number_of_predictions: int = MAX_NUMBER_OF_PREDICTIONS,
        message_preprocessor: Optional[LambdaType] = None,
        on_circuit_break: Optional[LambdaType] = None,
    ):
        self.interpreter = interpreter
        self.nlg = generator
        self.policy_ensemble = policy_ensemble
        self.domain = domain
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.max_number_of_predictions = max_number_of_predictions
        self.message_preprocessor = message_preprocessor
        self.on_circuit_break = on_circuit_break
        self.action_endpoint = action_endpoint

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message with this processor."""

        # preprocess message if necessary
        tracker = await self.log_message(message, should_save_tracker=False)

        if not self.policy_ensemble or not self.domain:
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)
            rasa.shared.utils.io.raise_warning(
                "No policy ensemble or domain set. Skipping action prediction "
                "and execution.",
                docs=DOCS_URL_POLICIES,
            )
            return None

        await self._predict_and_execute_next_action(message.output_channel, tracker)

        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return message.output_channel.messages

        return None

    async def predict_next(self, sender_id: Text) -> Optional[Dict[Text, Any]]:
        """Predict the next action for the current conversation state.

        Args:
            sender_id: Conversation ID.

        Returns:
            The prediction for the next action. `None` if no domain or policies loaded.
        """
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = await self.fetch_tracker_and_update_session(sender_id)
        result = self.predict_next_with_tracker(tracker)

        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        return result

    def predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        verbosity: EventVerbosity = EventVerbosity.AFTER_RESTART,
    ) -> Optional[Dict[Text, Any]]:
        """Predict the next action for a given conversation state.

        Args:
            tracker: A tracker representing a conversation state.
            verbosity: Verbosity for the returned conversation state.

        Returns:
            The prediction for the next action. `None` if no domain or policies loaded.
        """
        if not self.policy_ensemble or not self.domain:
            # save tracker state to continue conversation from this state
            rasa.shared.utils.io.raise_warning(
                "No policy ensemble or domain set. Skipping action prediction."
                "You should set a policy before training a model.",
                docs=DOCS_URL_POLICIES,
            )
            return None

        prediction = self._get_next_action_probabilities(tracker)

        scores = [
            {"action": a, "score": p}
            for a, p in zip(self.domain.action_names_or_texts, prediction.probabilities)
        ]
        return {
            "scores": scores,
            "policy": prediction.policy_name,
            "confidence": prediction.max_confidence,
            "tracker": tracker.current_state(verbosity),
        }

    async def _update_tracker_session(
        self,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Check the current session in `tracker` and update it if expired.

        An 'action_session_start' is run if the latest tracker session has expired,
        or if the tracker does not yet contain any events (only those after the last
        restart are considered).

        Args:
            metadata: Data sent from client associated with the incoming user message.
            tracker: Tracker to inspect.
            output_channel: Output channel for potential utterances in a custom
                `ActionSessionStart`.
        """
        if not tracker.applied_events() or self._has_session_expired(tracker):
            logger.debug(
                f"Starting a new session for conversation ID '{tracker.sender_id}'."
            )

            action_session_start = self._get_action(ACTION_SESSION_START_NAME)
            # TODO: Remove in 3.0.0 and describe migration to `session_start_metadata`
            # slot in migration guide.
            if isinstance(
                action_session_start, rasa.core.actions.action.ActionSessionStart
            ):
                # Here we set optional metadata to the ActionSessionStart, which will
                # then be passed to the SessionStart event.
                action_session_start.metadata = metadata

            if metadata:
                tracker.update(
                    SlotSet(SESSION_START_METADATA_SLOT, metadata), self.domain
                )

            await self._run_action(
                action=action_session_start,
                tracker=tracker,
                output_channel=output_channel,
                nlg=self.nlg,
                prediction=PolicyPrediction.for_action_name(
                    self.domain, ACTION_SESSION_START_NAME
                ),
            )

    async def fetch_tracker_and_update_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Fetches tracker for `sender_id` and updates its conversation session.

        If a new tracker is created, `action_session_start` is run.

        Args:
            metadata: Data sent from client associated with the incoming user message.
            output_channel: Output channel associated with the incoming user message.
            sender_id: Conversation ID for which to fetch the tracker.

        Returns:
              Tracker for `sender_id`.
        """
        tracker = self.get_tracker(sender_id)

        await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def fetch_tracker_with_initial_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Fetches tracker for `sender_id` and runs a session start if it's a new
        tracker.

        Args:
            metadata: Data sent from client associated with the incoming user message.
            output_channel: Output channel associated with the incoming user message.
            sender_id: Conversation ID for which to fetch the tracker.

        Returns:
              Tracker for `sender_id`.
        """
        tracker = self.get_tracker(sender_id)

        # run session start only if the tracker is empty
        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    def get_tracker(self, conversation_id: Text) -> DialogueStateTracker:
        """Get the tracker for a conversation.

        In contrast to `fetch_tracker_and_update_session` this does not add any
        `action_session_start` or `session_start` events at the beginning of a
        conversation.

        Args:
            conversation_id: The ID of the conversation for which the history should be
                retrieved.

        Returns:
            Tracker for the conversation. Creates an empty tracker in case it's a new
            conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        return self.tracker_store.get_or_create_tracker(
            conversation_id, append_action_listen=False
        )

    def get_trackers_for_all_conversation_sessions(
        self, conversation_id: Text
    ) -> List[DialogueStateTracker]:
        """Fetches all trackers for a conversation.

        Individual trackers are returned for each conversation session found
        for `conversation_id`.

        Args:
            conversation_id: The ID of the conversation for which the trackers should
                be retrieved.

        Returns:
            Trackers for the conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = self.tracker_store.retrieve_full_tracker(conversation_id)

        return rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)

    async def log_message(
        self, message: UserMessage, should_save_tracker: bool = True
    ) -> DialogueStateTracker:
        """Log `message` on tracker belonging to the message's conversation_id.

        Optionally save the tracker if `should_save_tracker` is `True`. Tracker saving
        can be skipped if the tracker returned by this method is used for further
        processing and saved at a later stage.
        """
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = await self.fetch_tracker_and_update_session(
            message.sender_id, message.output_channel, message.metadata
        )

        await self._handle_message_with_tracker(message, tracker)

        if should_save_tracker:
            # save tracker state to continue conversation from this state
            self._save_tracker(tracker)

        return tracker

    async def execute_action(
        self,
        sender_id: Text,
        action_name: Text,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> Optional[DialogueStateTracker]:
        """Execute an action for a conversation.

        Note that this might lead to unexpected bot behavior. Rather use an intent
        to execute certain behavior within a conversation (e.g. by using
        `trigger_external_user_uttered`).

        Args:
            sender_id: The ID of the conversation.
            action_name: The name of the action which should be executed.
            output_channel: The output channel which should be used for bot responses.
            nlg: The response generator.
            prediction: The prediction for the action.

        Returns:
            The new conversation state. Note that the new state is also persisted.
        """
        # we have a Tracker instance for each user
        # which maintains conversation state
        tracker = await self.fetch_tracker_and_update_session(sender_id, output_channel)

        action = self._get_action(action_name)
        await self._run_action(action, tracker, output_channel, nlg, prediction)

        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

        return tracker

    def predict_next_action(
        self, tracker: DialogueStateTracker
    ) -> Tuple[rasa.core.actions.action.Action, PolicyPrediction]:
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use
        ML to predict the action. Returns the index of the next action.
        """
        prediction = self._get_next_action_probabilities(tracker)

        action = rasa.core.actions.action.action_for_index(
            prediction.max_confidence_index, self.domain, self.action_endpoint
        )

        logger.debug(
            f"Predicted next action '{action.name()}' with confidence "
            f"{prediction.max_confidence:.2f}."
        )

        return action, prediction

    @staticmethod
    def _is_reminder(e: Event, name: Text) -> bool:
        return isinstance(e, ReminderScheduled) and e.name == name

    @staticmethod
    def _is_reminder_still_valid(
        tracker: DialogueStateTracker, reminder_event: ReminderScheduled
    ) -> bool:
        """Check if the conversation has been restarted after reminder."""

        for e in reversed(tracker.applied_events()):
            if MessageProcessor._is_reminder(e, reminder_event.name):
                return True
        return False  # not found in applied events --> has been restarted

    @staticmethod
    def _has_message_after_reminder(
        tracker: DialogueStateTracker, reminder_event: ReminderScheduled
    ) -> bool:
        """Check if the user sent a message after the reminder."""

        for e in reversed(tracker.events):
            if MessageProcessor._is_reminder(e, reminder_event.name):
                return False

            if isinstance(e, UserUttered) and e.text:
                return True

        return True  # tracker has probably been restarted

    async def handle_reminder(
        self,
        reminder_event: ReminderScheduled,
        sender_id: Text,
        output_channel: OutputChannel,
    ) -> None:
        """Handle a reminder that is triggered asynchronously."""
        async with self.lock_store.lock(sender_id):
            tracker = await self.fetch_tracker_and_update_session(
                sender_id, output_channel
            )

            if (
                reminder_event.kill_on_user_message
                and self._has_message_after_reminder(tracker, reminder_event)
                or not self._is_reminder_still_valid(tracker, reminder_event)
            ):
                logger.debug(
                    f"Canceled reminder because it is outdated ({reminder_event})."
                )
            else:
                intent = reminder_event.intent
                entities = reminder_event.entities or {}
                await self.trigger_external_user_uttered(
                    intent, entities, tracker, output_channel
                )

    async def trigger_external_user_uttered(
        self,
        intent_name: Text,
        entities: Optional[Union[List[Dict[Text, Any]], Dict[Text, Text]]],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Triggers an external message.

        Triggers an external message (like a user message, but invisible;
        used, e.g., by a reminder or the trigger_intent endpoint).

        Args:
            intent_name: Name of the intent to be triggered.
            entities: Entities to be passed on.
            tracker: The tracker to which the event should be added.
            output_channel: The output channel.
        """
        if isinstance(entities, list):
            entity_list = entities
        elif isinstance(entities, dict):
            # Allow for a short-hand notation {"ent1": "val1", "ent2": "val2", ...}.
            # Useful if properties like 'start', 'end', or 'extractor' are not given,
            # e.g. for external events.
            entity_list = [
                {"entity": ent, "value": val} for ent, val in entities.items()
            ]
        elif not entities:
            entity_list = []
        else:
            rasa.shared.utils.io.raise_warning(
                f"Invalid entity specification: {entities}. Assuming no entities."
            )
            entity_list = []

        # Set the new event's input channel to the latest input channel, so
        # that we don't lose this property.
        input_channel = tracker.get_latest_input_channel()

        tracker.update(
            UserUttered.create_external(intent_name, entity_list, input_channel),
            self.domain,
        )
        await self._predict_and_execute_next_action(output_channel, tracker)
        # save tracker state to continue conversation from this state
        self._save_tracker(tracker)

    @staticmethod
    def _log_slots(tracker: DialogueStateTracker) -> None:
        # Log currently set slots
        slot_values = "\n".join(
            [f"\t{s.name}: {s.value}" for s in tracker.slots.values()]
        )
        if slot_values.strip():
            logger.debug(f"Current slot values: \n{slot_values}")

    def _check_for_unseen_features(self, parse_data: Dict[Text, Any]) -> None:
        """Warns the user if the NLU parse data contains unrecognized features.

        Checks intents and entities picked up by the NLU interpreter
        against the domain and warns the user of those that don't match.
        Also considers a list of default intents that are valid but don't
        need to be listed in the domain.

        Args:
            parse_data: NLUInterpreter parse data to check against the domain.
        """
        if not self.domain or self.domain.is_empty():
            return

        intent = parse_data["intent"][INTENT_NAME_KEY]
        if intent and intent not in self.domain.intents:
            rasa.shared.utils.io.raise_warning(
                f"Interpreter parsed an intent '{intent}' "
                f"which is not defined in the domain. "
                f"Please make sure all intents are listed in the domain.",
                docs=DOCS_URL_DOMAINS,
            )

        entities = parse_data["entities"] or []
        for element in entities:
            entity = element["entity"]
            if entity and entity not in self.domain.entities:
                rasa.shared.utils.io.raise_warning(
                    f"Interpreter parsed an entity '{entity}' "
                    f"which is not defined in the domain. "
                    f"Please make sure all entities are listed in the domain.",
                    docs=DOCS_URL_DOMAINS,
                )

    def _get_action(
        self, action_name: Text
    ) -> Optional[rasa.core.actions.action.Action]:
        return rasa.core.actions.action.action_for_name_or_text(
            action_name, self.domain, self.action_endpoint
        )

    async def parse_message(
        self, message: UserMessage, tracker: Optional[DialogueStateTracker] = None
    ) -> Dict[Text, Any]:
        """Interprete the passed message using the NLU interpreter.

        Arguments:
            message: Message to handle
            tracker: Dialogue context of the message

        Returns:
            Parsed data extracted from the message.
        """
        # preprocess message if necessary
        if self.message_preprocessor is not None:
            text = self.message_preprocessor(message.text)
        else:
            text = message.text

        # for testing - you can short-cut the NLU part with a message
        # in the format /intent{"entity1": val1, "entity2": val2}
        # parse_data is a dict of intent & entities
        if text.startswith(INTENT_MESSAGE_PREFIX):
            parse_data = await RegexInterpreter().parse(
                text, message.message_id, tracker
            )
        else:
            parse_data = await self.interpreter.parse(
                text, message.message_id, tracker, metadata=message.metadata
            )

        logger.debug(
            "Received user message '{}' with intent '{}' "
            "and entities '{}'".format(
                message.text, parse_data["intent"], parse_data["entities"]
            )
        )

        self._check_for_unseen_features(parse_data)

        return parse_data

    async def _handle_message_with_tracker(
        self, message: UserMessage, tracker: DialogueStateTracker
    ) -> None:

        if message.parse_data:
            parse_data = message.parse_data
        else:
            parse_data = await self.parse_message(message, tracker)

        # don't ever directly mutate the tracker
        # - instead pass its events to log
        tracker.update(
            UserUttered(
                message.text,
                parse_data["intent"],
                parse_data["entities"],
                parse_data,
                input_channel=message.input_channel,
                message_id=message.message_id,
                metadata=message.metadata,
            ),
            self.domain,
        )

        if parse_data["entities"]:
            self._log_slots(tracker)

        logger.debug(
            f"Logged UserUtterance - tracker now has {len(tracker.events)} events."
        )

    @staticmethod
    def _should_handle_message(tracker: DialogueStateTracker) -> bool:
        return (
            not tracker.is_paused()
            or tracker.latest_message.intent.get(INTENT_NAME_KEY) == USER_INTENT_RESTART
        )

    def is_action_limit_reached(
        self, num_predicted_actions: int, should_predict_another_action: bool
    ) -> bool:
        """Check whether the maximum number of predictions has been met.

        Args:
            num_predicted_actions: Number of predicted actions.
            should_predict_another_action: Whether the last executed action allows
            for more actions to be predicted or not.

        Returns:
            `True` if the limit of actions to predict has been reached.
        """
        return (
            num_predicted_actions >= self.max_number_of_predictions
            and should_predict_another_action
        )

    async def _predict_and_execute_next_action(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> None:
        # keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True
        num_predicted_actions = 0

        # action loop. predicts actions until we hit action listen
        while (
            should_predict_another_action
            and self._should_handle_message(tracker)
            and num_predicted_actions < self.max_number_of_predictions
        ):
            # this actually just calls the policy's method by the same name
            action, prediction = self.predict_next_action(tracker)

            should_predict_another_action = await self._run_action(
                action, tracker, output_channel, self.nlg, prediction
            )
            num_predicted_actions += 1

        if self.is_action_limit_reached(
            num_predicted_actions, should_predict_another_action
        ):
            # circuit breaker was tripped
            logger.warning(
                "Circuit breaker tripped. Stopped predicting "
                f"more actions for sender '{tracker.sender_id}'."
            )
            if self.on_circuit_break:
                # call a registered callback
                self.on_circuit_break(tracker, output_channel, self.nlg)

    @staticmethod
    def should_predict_another_action(action_name: Text) -> bool:
        """Determine whether the processor should predict another action.

        Args:
            action_name: Name of the latest executed action.

        Returns:
            `False` if `action_name` is `ACTION_LISTEN_NAME` or
            `ACTION_SESSION_START_NAME`, otherwise `True`.
        """

        return action_name not in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME)

    async def execute_side_effects(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Send bot messages, schedule and cancel reminders that are logged
        in the events array."""

        await self._send_bot_messages(events, tracker, output_channel)
        await self._schedule_reminders(events, tracker, output_channel)
        await self._cancel_reminders(events, tracker)

    @staticmethod
    async def _send_bot_messages(
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Send all the bot messages that are logged in the events array."""

        for e in events:
            if not isinstance(e, BotUttered):
                continue

            await output_channel.send_response(tracker.sender_id, e.message())

    async def _schedule_reminders(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> None:
        """Uses the scheduler to time a job to trigger the passed reminder.

        Reminders with the same `id` property will overwrite one another
        (i.e. only one of them will eventually run).
        """
        for e in events:
            if not isinstance(e, ReminderScheduled):
                continue

            (await jobs.scheduler()).add_job(
                self.handle_reminder,
                "date",
                run_date=e.trigger_date_time,
                args=[e, tracker.sender_id, output_channel],
                id=e.name,
                replace_existing=True,
                name=e.scheduled_job_name(tracker.sender_id),
            )

    @staticmethod
    async def _cancel_reminders(
        events: List[Event], tracker: DialogueStateTracker
    ) -> None:
        """Cancel reminders that match the `ReminderCancelled` event."""
        # All Reminders specified by ReminderCancelled events will be cancelled
        for event in events:
            if isinstance(event, ReminderCancelled):
                scheduler = await jobs.scheduler()
                for scheduled_job in scheduler.get_jobs():
                    if event.cancels_job_with_name(
                        scheduled_job.name, tracker.sender_id
                    ):
                        scheduler.remove_job(scheduled_job.id)

    async def _run_action(
        self,
        action: rasa.core.actions.action.Action,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        prediction: PolicyPrediction,
    ) -> bool:
        # events and return values are used to update
        # the tracker state after an action has been taken
        try:
            # Use temporary tracker as we might need to discard the policy events in
            # case of a rejection.
            temporary_tracker = tracker.copy()
            temporary_tracker.update_with_events(prediction.events, self.domain)
            events = await action.run(
                output_channel, nlg, temporary_tracker, self.domain
            )
        except rasa.core.actions.action.ActionExecutionRejection:
            events = [
                ActionExecutionRejected(
                    action.name(), prediction.policy_name, prediction.max_confidence
                )
            ]
            tracker.update(events[0])
            return self.should_predict_another_action(action.name())
        except Exception:
            logger.exception(
                f"Encountered an exception while running action '{action.name()}'."
                "Bot will continue, but the actions events are lost. "
                "Please check the logs of your action server for "
                "more information."
            )
            events = []

        self._log_action_on_tracker(tracker, action, events, prediction)
        if action.name() != ACTION_LISTEN_NAME and not action.name().startswith(
            UTTER_PREFIX
        ):
            self._log_slots(tracker)

        await self.execute_side_effects(events, tracker, output_channel)

        return self.should_predict_another_action(action.name())

    def _warn_about_new_slots(
        self, tracker: DialogueStateTracker, action_name: Text, events: List[Event]
    ) -> None:
        # these are the events from that action we have seen during training

        if (
            not self.policy_ensemble
            or action_name not in self.policy_ensemble.action_fingerprints
        ):
            return

        fingerprint = self.policy_ensemble.action_fingerprints[action_name]
        slots_seen_during_train = fingerprint.get(SLOTS, set())
        for e in events:
            if isinstance(e, SlotSet) and e.key not in slots_seen_during_train:
                s: Optional[Slot] = tracker.slots.get(e.key)
                if s and s.has_features():
                    if e.key == REQUESTED_SLOT and tracker.active_loop:
                        pass
                    else:
                        rasa.shared.utils.io.raise_warning(
                            f"Action '{action_name}' set slot type '{s.type_name}' "
                            f"which it never set during the training. This "
                            f"can throw off the prediction. Make sure to "
                            f"include training examples in your stories "
                            f"for the different types of slots this "
                            f"action can return. Remember: you need to "
                            f"set the slots manually in the stories by "
                            f"adding the following lines after the action:\n\n"
                            f"- {KEY_ACTION}: {action_name}\n"
                            f"- {KEY_SLOT_NAME}:\n"
                            f"  - {e.key}: {e.value}\n",
                            docs=DOCS_URL_SLOTS,
                        )

    def _log_action_on_tracker(
        self,
        tracker: DialogueStateTracker,
        action: Action,
        events: Optional[List[Event]],
        prediction: PolicyPrediction,
    ) -> None:
        # Ensures that the code still works even if a lazy programmer missed
        # to type `return []` at the end of an action or the run method
        # returns `None` for some other reason.
        if events is None:
            events = []

        self._warn_about_new_slots(tracker, action.name(), events)

        action_was_rejected_manually = any(
            isinstance(event, ActionExecutionRejected) for event in events
        )
        if not action_was_rejected_manually:
            logger.debug(f"Policy prediction ended with events '{prediction.events}'.")
            tracker.update_with_events(prediction.events, self.domain)

            # log the action and its produced events
            tracker.update(action.event_for_successful_execution(prediction))

        logger.debug(f"Action '{action.name()}' ended with events '{events}'.")
        tracker.update_with_events(events, self.domain)

    def _has_session_expired(self, tracker: DialogueStateTracker) -> bool:
        """Determine whether the latest session in `tracker` has expired.

        Args:
            tracker: Tracker to inspect.

        Returns:
            `True` if the session in `tracker` has expired, `False` otherwise.
        """
        if not self.domain.session_config.are_sessions_enabled():
            # tracker has never expired if sessions are disabled
            return False

        user_uttered_event: Optional[UserUttered] = tracker.get_last_event_for(
            UserUttered
        )

        if not user_uttered_event:
            # there is no user event so far so the session should not be considered
            # expired
            return False

        time_delta_in_seconds = time.time() - user_uttered_event.timestamp
        has_expired = (
            time_delta_in_seconds / 60
            > self.domain.session_config.session_expiration_time
        )
        if has_expired:
            logger.debug(
                f"The latest session for conversation ID '{tracker.sender_id}' has "
                f"expired."
            )

        return has_expired

    def _save_tracker(self, tracker: DialogueStateTracker) -> None:
        self.tracker_store.save(tracker)

    def _get_next_action_probabilities(
        self, tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        """Collect predictions from ensemble and return action and predictions."""
        followup_action = tracker.followup_action
        if followup_action:
            tracker.clear_followup_action()
            if followup_action in self.domain.action_names_or_texts:
                return PolicyPrediction.for_action_name(
                    self.domain, followup_action, FOLLOWUP_ACTION
                )

            logger.error(
                f"Trying to run unknown follow-up action '{followup_action}'. "
                "Instead of running that, Rasa Open Source will ignore the action "
                "and predict the next action."
            )

        prediction = self.policy_ensemble.probabilities_using_best_policy(
            tracker, self.domain, self.interpreter
        )

        if isinstance(prediction, PolicyPrediction):
            return prediction

        rasa.shared.utils.io.raise_deprecation_warning(
            f"Returning a tuple of probabilities and policy name for "
            f"`{PolicyEnsemble.probabilities_using_best_policy.__name__}` is "
            f"deprecated and will be removed in Rasa Open Source 3.0.0. Please return "
            f"a `{PolicyPrediction.__name__}` object instead."
        )
        probabilities, policy_name = prediction
        return PolicyPrediction(probabilities, policy_name)
