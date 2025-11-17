import copy
import inspect
import os
import re
import tarfile
import time
from pathlib import Path
from types import LambdaType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text, Tuple, Union

import structlog

import rasa.core.actions.action
import rasa.core.tracker_stores.tracker_store
import rasa.core.utils
import rasa.shared.core.trackers
import rasa.shared.utils.io
from rasa.core import jobs
from rasa.core.actions.action import Action
from rasa.core.actions.action_exceptions import ActionExecutionRejection
from rasa.core.actions.forms import FormAction
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    OutputChannel,
    UserMessage,
)
from rasa.core.constants import KEY_IS_CALM_SYSTEM, KEY_IS_COEXISTENCE_ASSISTANT
from rasa.core.http_interpreter import RasaNLUHttpInterpreter
from rasa.core.lock_store import LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.policy import PolicyPrediction
from rasa.dialogue_understanding.commands import (
    CannotHandleCommand,
    Command,
    NoopCommand,
    RestartCommand,
    SessionEndCommand,
    SessionStartCommand,
    SetSlotCommand,
)
from rasa.dialogue_understanding.commands.utils import (
    create_validate_frames_from_slot_set_events,
)
from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.dialogue_understanding.utils import add_commands_to_message_parse_data
from rasa.engine import loader
from rasa.engine.constants import (
    PLACEHOLDER_ENDPOINTS,
    PLACEHOLDER_MESSAGE,
    PLACEHOLDER_OUTPUT_CHANNEL,
    PLACEHOLDER_TRACKER,
)
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelMetadata
from rasa.exceptions import ActionLimitReached, ModelNotFound
from rasa.model import get_latest_model
from rasa.plugin import plugin_manager
from rasa.shared.constants import (
    ASSISTANT_ID_KEY,
    DEFAULT_SENDER_ID,
    DOCS_URL_DOMAINS,
    DOCS_URL_NLU_BASED_POLICIES,
    RASA_PATTERN_CANNOT_HANDLE_INVALID_INTENT,
    ROUTE_TO_CALM_SLOT,
    UTTER_PREFIX,
)
from rasa.shared.core.constants import (
    ACTION_AGENT_REQUEST_USER_INPUT_NAME,
    ACTION_CORRECT_FLOW_SLOT,
    ACTION_EXTRACT_SLOTS,
    ACTION_LISTEN_NAME,
    ACTION_METADATA_EXECUTION_TIME,
    ACTION_SESSION_START_NAME,
    FOLLOWUP_ACTION,
    SESSION_START_METADATA_SLOT,
    SILENCE_TIMEOUT_SLOT,
    SLOT_CONSECUTIVE_SILENCE_TIMEOUTS,
    USER_INTENT_RESTART,
    USER_INTENT_SILENCE_TIMEOUT,
    SetSlotExtractor,
)
from rasa.shared.core.events import (
    ActionExecuted,
    ActionExecutionRejected,
    BotUttered,
    Event,
    ReminderCancelled,
    ReminderScheduled,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.data import TrainingType, create_regex_pattern_reader
from rasa.shared.nlu.constants import (
    COMMANDS,
    ENTITIES,
    FULL_RETRIEVAL_INTENT_NAME_KEY,
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RESPONSE_KEY,
    PREDICTED_COMMANDS,
    PREDICTED_CONFIDENCE_KEY,
    RESPONSE,
    RESPONSE_SELECTOR,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.common import TempDirectoryPath, get_temp_dir_name
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from rasa.core.config.available_endpoints import AvailableEndpoints
    from rasa.privacy.privacy_manager import BackgroundPrivacyManager

structlogger = structlog.get_logger()

MAX_NUMBER_OF_PREDICTIONS = int(os.environ.get("MAX_NUMBER_OF_PREDICTIONS", "10"))
MAX_NUMBER_OF_PREDICTIONS_CALM = int(
    os.environ.get("MAX_NUMBER_OF_PREDICTIONS_CALM", "1000")
)


class MessageProcessor:
    """The message processor is interface for communicating with a bot model."""

    def __init__(
        self,
        model_path: Union[Text, Path],
        tracker_store: rasa.core.tracker_stores.tracker_store.TrackerStore,
        lock_store: LockStore,
        generator: NaturalLanguageGenerator,
        action_endpoint: Optional[EndpointConfig] = None,
        max_number_of_predictions: int = MAX_NUMBER_OF_PREDICTIONS,
        max_number_of_predictions_calm: int = MAX_NUMBER_OF_PREDICTIONS_CALM,
        on_circuit_break: Optional[LambdaType] = None,
        http_interpreter: Optional[RasaNLUHttpInterpreter] = None,
        endpoints: Optional["AvailableEndpoints"] = None,
        privacy_manager: Optional["BackgroundPrivacyManager"] = None,
    ) -> None:
        """Initializes a `MessageProcessor`."""
        self.nlg = generator
        self.tracker_store = tracker_store
        self.lock_store = lock_store
        self.on_circuit_break = on_circuit_break
        self.action_endpoint = action_endpoint
        self.model_filename, self.model_metadata, self.graph_runner = self._load_model(
            model_path
        )
        self.endpoints = endpoints

        self.max_number_of_predictions = max_number_of_predictions
        self.max_number_of_predictions_calm = max_number_of_predictions_calm
        self.is_calm_assistant = self._is_calm_assistant()

        if self.model_metadata.assistant_id is None:
            rasa.shared.utils.io.raise_warning(
                f"The model metadata does not contain a value for the "
                f"'{ASSISTANT_ID_KEY}' attribute. Check that 'config.yml' "
                f"file contains a value for the '{ASSISTANT_ID_KEY}' key "
                f"and re-train the model. Failure to do so will result in "
                f"streaming events without a unique assistant identifier.",
                UserWarning,
            )

        self.model_path = Path(model_path)
        self.domain = self.model_metadata.domain
        self.http_interpreter = http_interpreter
        self.privacy_manager = privacy_manager
        if self.privacy_manager is not None:
            self.privacy_manager.validate_sensitive_slots_in_domain(self.domain)

    @staticmethod
    def _load_model(
        model_path: Union[Text, Path],
    ) -> Tuple[Text, ModelMetadata, GraphRunner]:
        """Unpacks a model from a given path using the graph model loader."""
        try:
            if os.path.isfile(model_path):
                model_tar = model_path
            else:
                model_file_path = get_latest_model(model_path)
                if not model_file_path:
                    raise ModelNotFound(f"No model found at path '{model_path}'.")
                model_tar = model_file_path
        except TypeError:
            raise ModelNotFound(f"Model {model_path} can not be loaded.")

        structlogger.info(
            "rasa.core.processor.load_model",
            event_info="Loading model.",
            model_path=model_tar,
        )
        with TempDirectoryPath(get_temp_dir_name()) as temporary_directory:
            try:
                metadata, runner = loader.load_predict_graph_runner(
                    Path(temporary_directory),
                    Path(model_tar),
                    LocalModelStorage,
                    DaskGraphRunner,
                )
                return os.path.basename(model_tar), metadata, runner
            except tarfile.ReadError:
                raise ModelNotFound(f"Model {model_path} can not be loaded.")

    async def handle_message(
        self, message: UserMessage
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message with this processor."""
        # preprocess message if necessary
        self.time_turn_start = time.time()
        tracker = await self.log_message(message, should_save_tracker=False)

        if self.model_metadata.training_type == TrainingType.NLU:
            await self.save_tracker(tracker)
            rasa.shared.utils.io.raise_warning(
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_NLU_BASED_POLICIES,
            )
            return None

        tracker = await self.run_action_extract_slots(message.output_channel, tracker)

        await self._run_prediction_loop(message.output_channel, tracker)

        await self.save_tracker(tracker)

        self.trigger_anonymization(tracker)

        if isinstance(message.output_channel, CollectingOutputChannel):
            return message.output_channel.messages

        return None

    def trigger_anonymization(self, tracker: DialogueStateTracker) -> None:
        if self.privacy_manager is None:
            structlogger.debug(
                "processor.trigger_anonymization.skipping.pii_management_not_enabled",
            )
            return None

        if not self.privacy_manager.event_brokers:
            structlogger.debug(
                "processor.trigger_anonymization.skipping.no_event_brokers",
            )
            return None

        structlogger.info(
            "rasa.core.processor.trigger_anonymization",
            sender_id=tracker.sender_id,
            event_info="Triggering anonymization for publishing anonymized "
            "events to the event broker.",
        )
        return self.privacy_manager.run(tracker)

    async def run_action_extract_slots(
        self,
        output_channel: OutputChannel,
        tracker: DialogueStateTracker,
    ) -> DialogueStateTracker:
        """Run action to extract slots and update the tracker accordingly.

        Args:
            output_channel: Output channel associated with the incoming user message.
            tracker: A tracker representing a conversation state.

        Returns:
            the given (updated) tracker
        """
        action_extract_slots = rasa.core.actions.action.action_for_name_or_text(
            ACTION_EXTRACT_SLOTS, self.domain, self.action_endpoint
        )
        metadata = await self._add_flows_to_metadata()
        metadata[KEY_IS_CALM_SYSTEM] = self.message_contains_commands(
            tracker.latest_message
        )
        metadata[KEY_IS_COEXISTENCE_ASSISTANT] = self._is_coexistence_assistant(tracker)

        extraction_events = await action_extract_slots.run(
            output_channel, self.nlg, tracker, self.domain, metadata
        )

        await self._send_bot_messages(extraction_events, tracker, output_channel)

        tracker.update_with_events(extraction_events)

        structlogger.debug(
            "processor.extract.slots",
            action_extract_slot=ACTION_EXTRACT_SLOTS,
            len_extraction_events=len(extraction_events),
            rasa_events=copy.deepcopy(extraction_events),
        )

        return tracker

    async def predict_next_for_sender_id(
        self, sender_id: Text
    ) -> Optional[Dict[Text, Any]]:
        """Predict the next action for the given sender_id.

        Args:
            sender_id: Conversation ID.

        Returns:
            The prediction for the next action. `None` if no domain or policies loaded.
        """
        tracker = await self.fetch_tracker_and_update_session(sender_id)
        result = await self.predict_next_with_tracker(tracker)

        # save tracker state to continue conversation from this state
        await self.save_tracker(tracker)

        return result

    async def predict_next_with_tracker(
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
        if self.model_metadata.training_type == TrainingType.NLU:
            rasa.shared.utils.io.raise_warning(
                "No core model. Skipping action prediction and execution.",
                docs=DOCS_URL_NLU_BASED_POLICIES,
            )
            return None

        prediction = await self._predict_next_with_tracker(tracker)

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
            structlogger.debug(
                "rasa.core.processor._update_tracker_session",
                event_info="Starting a new session.",
                sender_id=tracker.sender_id,
            )

            action_session_start = self._get_action(ACTION_SESSION_START_NAME)

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
        tracker = await self.get_tracker(sender_id)

        await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def fetch_tracker_with_initial_session(
        self,
        sender_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Fetches tracker for `sender_id` and runs a session start on a new one.

        Args:
            metadata: Data sent from client associated with the incoming user message.
            output_channel: Output channel associated with the incoming user message.
            sender_id: Conversation ID for which to fetch the tracker.

        Returns:
              Tracker for `sender_id`.
        """
        tracker = await self.get_tracker(sender_id)

        # run session start only if the tracker is empty
        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def get_tracker(self, conversation_id: Text) -> DialogueStateTracker:
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

        tracker = await self.tracker_store.get_or_create_tracker(
            conversation_id, append_action_listen=False
        )
        tracker.model_id = self.model_metadata.model_id
        if tracker.assistant_id is None:
            tracker.assistant_id = self.model_metadata.assistant_id
        return tracker

    async def fetch_full_tracker_with_initial_session(
        self,
        conversation_id: Text,
        output_channel: Optional[OutputChannel] = None,
        metadata: Optional[Dict] = None,
    ) -> DialogueStateTracker:
        """Get the full tracker for a conversation, including events after a restart.

        Args:
            conversation_id: The ID of the conversation for which the history should be
                retrieved.
            output_channel: Output channel associated with the incoming user message.
            metadata: Data sent from client associated with the incoming user message.

        Returns:
            Tracker for the conversation. Creates an empty tracker with a new session
            initialized in case it's a new conversation.
        """
        conversation_id = conversation_id or DEFAULT_SENDER_ID

        tracker = await self.tracker_store.get_or_create_full_tracker(
            conversation_id, False
        )
        tracker.model_id = self.model_metadata.model_id

        if tracker.assistant_id is None:
            tracker.assistant_id = self.model_metadata.assistant_id

        if not tracker.events:
            await self._update_tracker_session(tracker, output_channel, metadata)

        return tracker

    async def get_trackers_for_all_conversation_sessions(
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

        tracker = await self.tracker_store.retrieve_full_tracker(conversation_id)

        return rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)

    async def log_message(
        self, message: UserMessage, should_save_tracker: bool = True
    ) -> DialogueStateTracker:
        """Log `message` on tracker belonging to the message's conversation_id.

        Optionally save the tracker if `should_save_tracker` is `True`. Tracker saving
        can be skipped if the tracker returned by this method is used for further
        processing and saved at a later stage.
        """
        tracker = await self.fetch_tracker_and_update_session(
            message.sender_id, message.output_channel, message.metadata
        )

        await self._handle_message_with_tracker(message, tracker)

        if should_save_tracker:
            await self.save_tracker(tracker)

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
        await self.save_tracker(tracker)

        return tracker

    async def predict_next_with_tracker_if_should(
        self,
        tracker: DialogueStateTracker,
        output_channel: Optional[OutputChannel] = None,
    ) -> Tuple[rasa.core.actions.action.Action, PolicyPrediction]:
        """Predicts the next action the bot should take after seeing x.

        This should be overwritten by more advanced policies to use
        ML to predict the action.

        Returns:
             The index of the next action and prediction of the policy.

        Raises:
            ActionLimitReached if the limit of actions to predict has been reached.
        """
        should_predict_another_action = self.should_predict_another_action(
            tracker.latest_action_name
        )

        if self.is_action_limit_reached(tracker, should_predict_another_action):
            raise ActionLimitReached(
                "The limit of actions to predict has been reached."
            )

        prediction = await self._predict_next_with_tracker(tracker, output_channel)

        action = rasa.core.actions.action.action_for_index(
            prediction.max_confidence_index, self.domain, self.action_endpoint
        )

        structlogger.debug(
            "rasa.core.processor.predict_next_with_tracker_if_should",
            event_info="Predicted next action.",
            action=action.name(),
            confidence=prediction.max_confidence,
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
                structlogger.debug(
                    "rasa.core.processor.handle_reminder",
                    event_info="Canceled reminder because it is outdated.",
                    reminder_event=reminder_event,
                )
            else:
                intent = reminder_event.intent
                entities: Union[List[Dict], Dict] = reminder_event.entities or {}
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

        tracker = await self.run_action_extract_slots(output_channel, tracker)

        await self._run_prediction_loop(output_channel, tracker)
        # save tracker state to continue conversation from this state
        await self.save_tracker(tracker)

    @staticmethod
    def _log_slots(tracker: DialogueStateTracker) -> None:
        # Log currently set slots
        slots = {s.name: s.value for s in tracker.slots.values() if s.value is not None}

        structlogger.debug("processor.slots.log", slots=slots)

    def _check_for_unseen_features(self, parse_data: Dict[Text, Any]) -> None:
        """Warns the user if the NLU parse data contains unrecognized features.

        Checks intents and entities picked up by the NLU parsing
        against the domain and warns the user of those that don't match.
        Also considers a list of default intents that are valid but don't
        need to be listed in the domain.

        Args:
            parse_data: Message parse data to check against the domain.
        """
        if not self.domain or self.domain.is_empty():
            return

        intent = parse_data[INTENT][INTENT_NAME_KEY]
        if intent and intent not in self.domain.intents:
            rasa.shared.utils.io.raise_warning(
                f"Parsed an intent '{intent}' "
                f"which is not defined in the domain. "
                f"Please make sure all intents are listed in the domain.",
                docs=DOCS_URL_DOMAINS,
            )

        entities = parse_data[ENTITIES] or []
        for element in entities:
            entity = element["entity"]
            if entity and entity not in self.domain.entities:
                rasa.shared.utils.io.raise_warning(
                    f"Parsed an entity '{entity}' "
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
        self,
        message: UserMessage,
        tracker: Optional[DialogueStateTracker] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Interprets the passed message.

        Args:
            message: Message to handle.
            tracker: Tracker to use.
            only_output_properties: If `True`, restrict the output to
                Message.only_output_properties.

        Returns:
            Parsed data extracted from the message.
        """
        if self.http_interpreter:
            parse_data = await self.http_interpreter.parse(message)
        else:
            processed_message = Message({TEXT: message.text})

            all_flows = await self.get_flows()
            should_force_slot_command, slot_name = (
                rasa.core.utils.should_force_slot_filling(tracker, all_flows)
            )

            if should_force_slot_command:
                command = SetSlotCommand(
                    name=slot_name,
                    value=message.text,
                    extractor=SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
                )
                processed_message.set(COMMANDS, [command.as_dict()], add_to_output=True)
            else:
                regex_reader = create_regex_pattern_reader(message, self.domain)
                if regex_reader:
                    processed_message = regex_reader.unpack_regex_message(
                        message=processed_message, domain=self.domain
                    )

            # Invalid use of slash syntax, sanitize the message before passing
            # it to the graph
            if (
                processed_message.starts_with_slash_syntax()
                and not processed_message.has_intent()
                and not processed_message.has_commands()
            ):
                message = self._sanitize_message(message)

            # Intent or commands are not explicitly present. Pass message to graph.
            if not (processed_message.has_intent() or processed_message.has_commands()):
                parse_data = await self._parse_message_with_graph(
                    message, tracker, only_output_properties
                )

            # Intents or commands are presents. Bypasses the standard parsing
            # pipeline.
            else:
                parse_data = await self._parse_message_with_commands_and_intents(
                    processed_message, tracker, only_output_properties
                )

        self._update_full_retrieval_intent(parse_data)
        structlogger.debug(
            "processor.message.parse",
            parse_data_text=copy.deepcopy(parse_data[TEXT]),
            parse_data_intent=parse_data[INTENT],
            parse_data_entities=copy.deepcopy(parse_data[ENTITIES]),
        )

        self._check_for_unseen_features(parse_data)

        self._initialise_consecutive_silence_timeout_slots(parse_data, tracker)

        return parse_data

    def _sanitize_message(self, message: UserMessage) -> UserMessage:
        """Sanitize user message by removing prepended slashes before the content."""
        # Regex pattern to match leading slashes and any whitespace before
        # actual content
        pattern = r"^[/\s]+"
        # Remove the matched pattern from the beginning of the message
        message.text = re.sub(pattern, "", message.text).strip()
        return message

    async def _parse_message_with_commands_and_intents(
        self,
        message: Message,
        tracker: Optional[DialogueStateTracker] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Parses the message to handle commands or intent trigger."""
        parse_data: Dict[Text, Any] = {
            TEXT: "",
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
            ENTITIES: [],
        }
        parse_data.update(
            message.as_dict(only_output_properties=only_output_properties)
        )

        commands = parse_data.get(COMMANDS, [])

        # add commands from intent payloads
        if tracker and not commands:
            nlu_adapted_commands = await self._nlu_to_commands(parse_data, tracker)
            commands += nlu_adapted_commands

            if (
                tracker.has_coexistence_routing_slot
                and tracker.get_slot(ROUTE_TO_CALM_SLOT) is None
            ):
                # If we are currently not routing to either CALM or DM1:
                # - Sticky route to CALM if there are any commands
                #   from the trigger intent parsing
                # - Sticky route to DM1 if there are no commands present
                route_to_calm_slot_value = self._determine_route_to_calm_slot_value(
                    nlu_adapted_commands
                )
                commands += [
                    SetSlotCommand(
                        ROUTE_TO_CALM_SLOT, route_to_calm_slot_value
                    ).as_dict()
                ]

        parse_data[COMMANDS] = commands
        return parse_data

    def _determine_route_to_calm_slot_value(
        self, nlu_adapted_commands: List[Dict[str, Any]]
    ) -> Optional[bool]:
        """Determines what value should be assigned to `ROUTE_TO_CALM_SLOT`.

        Returns:
            - True: If any command other than:
                - SessionStartCommand
                - SessionEndCommand
                - RestartCommand
              is present.
            - None: If only ignored system commands are present.
            - False If no commands at all.
        """
        system_commands_to_ignore = [
            SessionStartCommand.command(),
            SessionEndCommand.command(),
            RestartCommand.command(),
        ]

        # Exclude the system commands, as it doesn't originate from the user's
        # input intent and shouldn't influence the decision for setting
        # ROUTE_TO_CALM_SLOT.
        intent_triggered_commands = [
            command
            for command in nlu_adapted_commands
            if command.get("command") not in system_commands_to_ignore
        ]

        if len(intent_triggered_commands) > 0:
            # There are commands other than system commands present - route to CALM
            return True
        elif len(nlu_adapted_commands) > 0:
            # Only system command is present — defer routing decision
            return None
        else:
            # No commands at all — route to DM1
            return False

    def _update_full_retrieval_intent(self, parse_data: Dict[Text, Any]) -> None:
        """Update the parse data with the full retrieval intent.

        Args:
            parse_data: Message parse data to update.
        """
        intent_name = parse_data.get(INTENT, {}).get(INTENT_NAME_KEY)
        response_selector = parse_data.get(RESPONSE_SELECTOR, {})
        all_retrieval_intents = response_selector.get("all_retrieval_intents", [])
        if intent_name and intent_name in all_retrieval_intents:
            retrieval_intent = (
                response_selector.get(intent_name, {})
                .get(RESPONSE, {})
                .get(INTENT_RESPONSE_KEY)
            )
            parse_data[INTENT][FULL_RETRIEVAL_INTENT_NAME_KEY] = retrieval_intent

    async def _nlu_to_commands(
        self, parse_data: Dict[str, Any], tracker: DialogueStateTracker
    ) -> List[Dict[str, Any]]:
        """Converts the NLU parse data to commands using the adaptor.

        This is used if we receive intents/entities directly using `/intent{...}`
        syntax. In this case, the nlu graph is not run. Therefore, we need to
        convert the parse data to commands outside the graph.
        """
        from rasa.dialogue_understanding.generator.nlu_command_adapter import (
            NLUCommandAdapter,
        )

        message = Message(parse_data)
        commands = NLUCommandAdapter.convert_nlu_to_commands(
            message, tracker, await self.get_flows(), self.domain
        )
        add_commands_to_message_parse_data(
            message, NLUCommandAdapter.__name__, commands
        )
        parse_data[PREDICTED_COMMANDS] = message.get(PREDICTED_COMMANDS, [])

        # if there are no converted commands and parsed data contains invalid intent
        # add CannotHandleCommand as fallback
        if len(commands) == 0 and self._contains_undefined_intent(Message(parse_data)):
            structlogger.warning(
                "processor.message.nlu_to_commands.invalid_intent",
                event_info=(
                    f"No NLU commands converted and parsed data contains"
                    f"invalid intent: {parse_data[INTENT]['name']}. "
                    f"Returning CannotHandleCommand() as a fallback."
                ),
                invalid_intent=parse_data[INTENT][INTENT_NAME_KEY],
            )
            commands.append(
                CannotHandleCommand(RASA_PATTERN_CANNOT_HANDLE_INVALID_INTENT)
            )

        return [command.as_dict() for command in commands]

    def _contains_undefined_intent(self, message: Message) -> bool:
        """Checks if the message contains an undefined intent."""
        intent_name = message.get(INTENT, {}).get(INTENT_NAME_KEY)
        return intent_name is not None and intent_name not in self.domain.intents

    async def _parse_message_with_graph(
        self,
        message: UserMessage,
        tracker: Optional[DialogueStateTracker] = None,
        only_output_properties: bool = True,
    ) -> Dict[Text, Any]:
        """Interprets the passed message.

        Arguments:
            message: Message to handle
            tracker: Tracker to use
            only_output_properties: If `True`, restrict the output to
                Message.only_output_properties.

        Returns:
            Parsed data extracted from the message.
        """
        results = await self.graph_runner.run(
            inputs={PLACEHOLDER_MESSAGE: [message], PLACEHOLDER_TRACKER: tracker},
            targets=[self.model_metadata.nlu_target],
        )
        parsed_messages = results[self.model_metadata.nlu_target]
        parsed_message = parsed_messages[0]
        parse_data = {
            TEXT: "",
            INTENT: {INTENT_NAME_KEY: None, PREDICTED_CONFIDENCE_KEY: 0.0},
            ENTITIES: [],
            COMMANDS: [],
        }
        parse_data.update(
            parsed_message.as_dict(only_output_properties=only_output_properties)
        )
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
                parse_data[INTENT],
                parse_data[ENTITIES],
                parse_data,
                input_channel=message.input_channel,
                message_id=message.message_id,
                metadata=message.metadata,
            ),
            self.domain,
        )

        if parse_data[ENTITIES]:
            self._log_slots(tracker)

        plugin_manager().hook.after_new_user_message(tracker=tracker)

        structlogger.debug(
            "rasa.core.processor.handle_message_with_tracker",
            event_info="Logged UserUtterance.",
            user_message=message.text,
            number_of_events=len(tracker.events),
        )

    @staticmethod
    def _should_handle_message(tracker: DialogueStateTracker) -> bool:
        return not tracker.is_paused() or MessageProcessor._last_user_intent_is_restart(
            tracker
        )

    @staticmethod
    def _last_user_intent_is_restart(tracker: DialogueStateTracker) -> bool:
        """Check if the last user intent is a restart intent."""
        return (
            tracker.latest_message is not None
            and tracker.latest_message.intent.get(INTENT_NAME_KEY)
            == USER_INTENT_RESTART
        )

    def _tracker_state_specific_action_limit(
        self, tracker: DialogueStateTracker
    ) -> int:
        """Select the action limit based on the tracker state.

        This function determines the maximum number of predictions that should be
        made during a dialogue conversation. Typically, the number of predictions
        is limited to the number of actions executed so far in the conversation.
        However, in certain states (e.g., when the user is correcting the
        conversation flow), more predictions may be allowed as the system traverses
        through a long dialogue flow.

        Additionally, if the `ROUTE_TO_CALM_SLOT` is present in the tracker slots,
        the action limit is adjusted to a separate limit for CALM-based flows.

        Args:
            tracker: instance of DialogueStateTracker.

        Returns:
        The maximum number of predictions to make.
        """
        # Check if it is a CALM assistant and if so, that the `ROUTE_TO_CALM_SLOT`
        # is either not present or set to `True`.
        # If it does, use the specific prediction limit for CALM assistants.
        # Otherwise, use the default prediction limit.
        if self.is_calm_assistant and (
            not tracker.has_coexistence_routing_slot
            or tracker.get_slot(ROUTE_TO_CALM_SLOT)
        ):
            max_number_of_predictions = self.max_number_of_predictions_calm
        else:
            max_number_of_predictions = self.max_number_of_predictions

        reversed_events = list(tracker.events)[::-1]
        is_conversation_in_flow_correction = False
        for e in reversed_events:
            if isinstance(e, ActionExecuted):
                if e.action_name in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME):
                    break
                elif e.action_name == ACTION_CORRECT_FLOW_SLOT:
                    is_conversation_in_flow_correction = True
                    break

        if is_conversation_in_flow_correction:
            # allow for more predictions to be made as we might be traversing through
            # a long flow. We multiply the number of predictions by 10 to allow for
            # more predictions to be made - the factor is a best guess.
            return max_number_of_predictions * 5

        # Return the default
        return max_number_of_predictions

    def is_action_limit_reached(
        self, tracker: DialogueStateTracker, should_predict_another_action: bool
    ) -> bool:
        """Check whether the maximum number of predictions has been met.

        Args:
            tracker: instance of DialogueStateTracker.
            should_predict_another_action: Whether the last executed action allows
            for more actions to be predicted or not.

        Returns:
            `True` if the limit of actions to predict has been reached.
        """
        reversed_events = list(tracker.events)[::-1]
        num_predicted_actions = 0
        state_specific_action_limit = self._tracker_state_specific_action_limit(tracker)

        for e in reversed_events:
            if isinstance(e, ActionExecuted):
                if e.action_name in (ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME):
                    break
                num_predicted_actions += 1

        return (
            num_predicted_actions >= state_specific_action_limit
            and should_predict_another_action
        )

    async def _run_prediction_loop(
        self, output_channel: OutputChannel, tracker: DialogueStateTracker
    ) -> None:
        # keep taking actions decided by the policy until it chooses to 'listen'
        should_predict_another_action = True

        tracker = await self.run_command_processor(tracker)
        self.time_command_processor = time.time()

        # action loop. predicts actions until we hit action listen
        while should_predict_another_action and self._should_handle_message(tracker):
            # this actually just calls the policy's method by the same name
            try:
                action, prediction = await self.predict_next_with_tracker_if_should(
                    tracker, output_channel
                )
            except ActionLimitReached:
                structlogger.warning(
                    "rasa.core.processor.run_prediction_loop",
                    event_info="Circuit breaker tripped. Stopped predicting more "
                    "actions.",
                    sender_id=tracker.sender_id,
                )
                if self.on_circuit_break:
                    # call a registered callback
                    self.on_circuit_break(tracker, output_channel, self.nlg)
                break

            if prediction.is_end_to_end_prediction:
                structlogger.debug(
                    "rasa.core.processor.run_prediction_loop",
                    event_info="An end-to-end prediction was made which has "
                    "triggered the 2nd execution of the default action.",
                    action=ACTION_EXTRACT_SLOTS,
                )
                tracker = await self.run_action_extract_slots(output_channel, tracker)

            should_predict_another_action = await self._run_action(
                action, tracker, output_channel, self.nlg, prediction
            )

    @staticmethod
    def should_predict_another_action(action_name: Text) -> bool:
        """Determine whether the processor should predict another action.

        Args:
            action_name: Name of the latest executed action.

        Returns:
            `False` if `action_name` is `ACTION_LISTEN_NAME` or
            `ACTION_SESSION_START_NAME`, otherwise `True`.
        """
        return action_name not in (
            ACTION_LISTEN_NAME,
            ACTION_SESSION_START_NAME,
            ACTION_AGENT_REQUEST_USER_INPUT_NAME,
        )

    async def execute_side_effects(
        self,
        events: List[Event],
        tracker: DialogueStateTracker,
        output_channel: Optional[OutputChannel],
    ) -> None:
        """Attach tracker, send bot messages, schedule and cancel reminders."""
        if output_channel:
            output_channel.attach_tracker_state(tracker)
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

    async def run_command_processor(
        self, tracker: DialogueStateTracker
    ) -> DialogueStateTracker:
        """Run the command processor to apply commands to the stack.

        The command processor applies all the commands from the NLU pipeline to the
        dialogue stack. The dialogue stack then acts as base for decision making for
        the policies that can use it.

        Args:
            tracker: the dialogue state tracker

        Returns:
        An updated tracker after commands have been applied
        """
        target = "command_processor"
        results = await self.graph_runner.run(
            inputs={PLACEHOLDER_TRACKER: tracker.copy()}, targets=[target]
        )
        events = results[target]
        tracker.update_with_events(events)
        return tracker

    async def get_flows(self) -> FlowsList:
        """Get the list of flows from the graph."""
        target = "flows_provider"
        results = await self.graph_runner.run(inputs={}, targets=[target])
        return results[target]

    async def _add_flows_to_metadata(self) -> Dict[Text, Any]:
        """Convert the flows to metadata."""
        flows = await self.get_flows()
        flows_metadata = {}
        for flow in flows.underlying_flows:
            flow_as_json = flow.as_json()
            flow_as_json.pop("id")
            flows_metadata[flow.id] = flow_as_json

        return {"all_flows": flows_metadata}

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
            validate_frames: List[ValidateSlotPatternFlowStackFrame] = []
            # check if the last action was a correction action
            # before validating the corrected slots
            if tracker.latest_action_name == ACTION_CORRECT_FLOW_SLOT:
                tracker, validate_frames = self.validate_corrected_slots(tracker)

            # Use temporary tracker as we might need to discard the policy events in
            # case of a rejection.
            temporary_tracker = tracker.copy()
            temporary_tracker.update_with_events(prediction.events)

            run_args = inspect.getfullargspec(action.run).args
            if "metadata" in run_args:
                metadata: Optional[Dict] = prediction.action_metadata

                if isinstance(action, FormAction):
                    flows_metadata = await self._add_flows_to_metadata()
                    flows_metadata[KEY_IS_CALM_SYSTEM] = self.message_contains_commands(
                        temporary_tracker.latest_message
                    )
                    flows_metadata[KEY_IS_COEXISTENCE_ASSISTANT] = (
                        self._is_coexistence_assistant(temporary_tracker)
                    )
                    metadata = prediction.action_metadata or {}
                    metadata.update(flows_metadata)

                events = await action.run(
                    output_channel,
                    nlg,
                    temporary_tracker,
                    self.domain,
                    metadata=metadata,
                )
            else:
                events = await action.run(
                    output_channel, nlg, temporary_tracker, self.domain
                )

            if validate_frames:
                stack = tracker.stack
                for frame in validate_frames:
                    stack.push(frame)
                new_events = tracker.create_stack_updated_events(stack)
                tracker.update_with_events(new_events)

            self._log_action_and_events_on_tracker(tracker, action, events, prediction)
        except ActionExecutionRejection:
            events = [
                ActionExecutionRejected(
                    action.name(), prediction.policy_name, prediction.max_confidence
                )
            ]
            tracker.update(events[0])
            return self.should_predict_another_action(action.name())
        except Exception as e:
            structlogger.exception(
                "rasa.core.processor.run_action.exception",
                event_info=f"Encountered an exception while "
                f"running action '{action.name()}'."
                f"Bot will continue, but the actions events are lost. "
                f"Please check the logs of your action server for "
                f"more information.",
            )
            error_messge = str(e)
            events = []
            self._log_action_prediction_on_tracker(
                tracker,
                action,
                prediction,
                was_successful=False,
                error_message=error_messge,
            )

        if any(isinstance(e, UserUttered) for e in events):
            structlogger.debug(
                "rasa.core.processor.run_action",
                message="A `UserUttered` event was returned by executing "
                f"action '{action.name()}'. This will run the default action "
                f"'{ACTION_EXTRACT_SLOTS}'.",
            )
            tracker = await self.run_action_extract_slots(output_channel, tracker)

        if action.name() != ACTION_LISTEN_NAME and not action.name().startswith(
            UTTER_PREFIX
        ):
            self._log_slots(tracker)

        await self.execute_side_effects(events, tracker, output_channel)
        plugin_manager().hook.after_action_executed(tracker=tracker)
        return self.should_predict_another_action(action.name())

    def _add_metadata_if_action_listen(
        self, action: Action, prediction: PolicyPrediction
    ) -> None:
        """Adds execution times to the ActionExecuted event metadata."""
        if not hasattr(self, "time_turn_start"):
            return

        if not hasattr(self, "time_command_processor"):
            return

        if not action.name() == ACTION_LISTEN_NAME:
            return

        if prediction.action_metadata is None:
            prediction.action_metadata = {}

        # calculate execution times
        execution_time_prediction_loop = (
            time.time() - self.time_command_processor
        ) * 1000
        execution_time_command_processor = (
            self.time_command_processor - self.time_turn_start
        ) * 1000
        prediction.action_metadata[ACTION_METADATA_EXECUTION_TIME] = {
            "command_processor": execution_time_command_processor,
            "prediction_loop": execution_time_prediction_loop,
        }

    def _log_action_and_events_on_tracker(
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

        action_was_rejected_manually = any(
            isinstance(event, ActionExecutionRejected) for event in events
        )
        if not action_was_rejected_manually:
            self._log_action_prediction_on_tracker(
                tracker, action, prediction, was_successful=True, error_message=None
            )

        structlogger.debug(
            "processor.actions.log",
            action_name=action.name(),
            rasa_events=copy.deepcopy(events),
        )
        tracker.update_with_events(events)

    def _log_action_prediction_on_tracker(
        self,
        tracker: DialogueStateTracker,
        action: Action,
        prediction: PolicyPrediction,
        was_successful: bool,
        error_message: Optional[str],
    ) -> None:
        structlogger.debug(
            "processor.actions.policy_prediction",
            prediction_events=copy.deepcopy(prediction.events),
            policy_name=prediction.policy_name,
            action_name=action.name(),
        )
        tracker.update_with_events(prediction.events)

        # log the action and its produced events
        self._add_metadata_if_action_listen(action, prediction)
        tracker.update(
            action.event_for_successful_execution(
                prediction, was_successful, error_message
            )
        )

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

        user_uttered_event = tracker.get_last_event_for(UserUttered)

        if not user_uttered_event or not isinstance(user_uttered_event, UserUttered):
            # there is no user event so far so the session should not be considered
            # expired
            return False

        time_delta_in_seconds = time.time() - user_uttered_event.timestamp
        has_expired = (
            time_delta_in_seconds / 60
            > self.domain.session_config.session_expiration_time
        )
        if has_expired:
            structlogger.debug(
                "rasa.core.processor.has_session_expired",
                event_info="The latest session has expired.",
                sender_id=tracker.sender_id,
            )

        return has_expired

    async def save_tracker(self, tracker: DialogueStateTracker) -> None:
        """Save the given tracker to the tracker store.

        Args:
            tracker: Tracker to be saved.
        """
        await self.tracker_store.save(tracker)

    async def _predict_next_with_tracker(
        self,
        tracker: DialogueStateTracker,
        output_channel: Optional[OutputChannel] = None,
    ) -> PolicyPrediction:
        """Collect predictions from ensemble and return action and predictions."""
        followup_action = tracker.followup_action
        if followup_action:
            tracker.clear_followup_action()
            if followup_action in self.domain.action_names_or_texts:
                prediction = PolicyPrediction.for_action_name(
                    self.domain, followup_action, FOLLOWUP_ACTION
                )
                return prediction

            structlogger.error(
                "rasa.core.processor.predict_next_with_tracker",
                event_info="Trying to run unknown follow-up action.",
                message=(
                    "Trying to run unknown follow-up action. Instead of running "
                    "that, Rasa Pro will ignore the action and predict the next action."
                ),
                followup_action=followup_action,
            )

        target = self.model_metadata.core_target
        if not target:
            raise ValueError("Cannot predict next action if there is no core target.")

        inputs: Dict[str, Any] = {
            PLACEHOLDER_TRACKER: tracker,
            PLACEHOLDER_ENDPOINTS: self.endpoints,
            PLACEHOLDER_OUTPUT_CHANNEL: output_channel,
        }

        results = await self.graph_runner.run(
            inputs=inputs,
            targets=[target],
        )
        policy_prediction = results[target]
        return policy_prediction

    @staticmethod
    def message_contains_commands(latest_message: Optional[UserUttered]) -> bool:
        """Check if the latest message contains commands."""
        if latest_message is None:
            return False

        commands = [
            Command.command_from_json(command) for command in latest_message.commands
        ]
        filtered_commands = [
            command
            for command in commands
            if not (
                isinstance(command, SetSlotCommand)
                and command.name == ROUTE_TO_CALM_SLOT
            )
            and not isinstance(command, NoopCommand)
        ]

        return len(filtered_commands) > 0

    def _is_calm_assistant(self) -> bool:
        """Inspects the nodes of the graph schema to decide if we are in CALM.

        To determine whether we are in CALM mode, we check if any node is
        associated with the `FlowPolicy`, which is indicative of a
        CALM assistant setup.

        Returns:
            bool: True if any node in the graph schema uses `FlowPolicy`.
        """
        flow_policy_class_path = "rasa.core.policies.flow_policy.FlowPolicy"
        return self._is_component_present_in_graph_nodes(flow_policy_class_path)

    @staticmethod
    def _is_coexistence_assistant(tracker: DialogueStateTracker) -> bool:
        """Inspect the tracker to decide if we are in coexistence.

        Returns:
            bool: True if the tracker contains the routine slot.
        """
        return tracker.slots.get(ROUTE_TO_CALM_SLOT) is not None

    def _is_component_present_in_graph_nodes(self, component_path: Text) -> bool:
        """Check if a component is present in the graph nodes.

        Args:
            component_path: The path of the component to check for.

        Returns:
            `True` if the component is present in the graph nodes, `False` otherwise.
        """
        # Get the graph schema's nodes from the graph runner.
        nodes: dict[str, Any] = self.graph_runner._graph_schema.nodes  # type: ignore[attr-defined]

        for node_name, schema_node in nodes.items():
            if (
                schema_node.uses is not None
                and f"{schema_node.uses.__module__}.{schema_node.uses.__name__}"
                == component_path
            ):
                return True

        return False

    def validate_corrected_slots(
        self,
        tracker: DialogueStateTracker,
    ) -> Tuple[DialogueStateTracker, List[ValidateSlotPatternFlowStackFrame]]:
        """Validate the slots that were corrected in the tracker."""
        prior_tracker_events = list(reversed(tracker.events))
        tracker, validate_frames = create_validate_frames_from_slot_set_events(
            tracker,
            prior_tracker_events,
            should_break=True,
        )

        return tracker, validate_frames

    @staticmethod
    def _initialise_consecutive_silence_timeout_slots(
        parse_data: Dict[str, Any],
        tracker: DialogueStateTracker,
    ) -> None:
        # resetting timeouts variables whenever something that is not a timeout occurs
        if (
            parse_data.get(INTENT, {}).get(INTENT_NAME_KEY)
            != USER_INTENT_SILENCE_TIMEOUT
            and tracker
        ):
            if (
                SLOT_CONSECUTIVE_SILENCE_TIMEOUTS in tracker.slots
                and tracker.slots[SLOT_CONSECUTIVE_SILENCE_TIMEOUTS].value != 0.0
            ):
                tracker.update(SlotSet(SLOT_CONSECUTIVE_SILENCE_TIMEOUTS, 0.0))
            if (
                SILENCE_TIMEOUT_SLOT in tracker.slots
                and tracker.slots[SILENCE_TIMEOUT_SLOT].value
                != tracker.slots[SILENCE_TIMEOUT_SLOT].initial_value
            ):
                tracker.update(
                    SlotSet(
                        SILENCE_TIMEOUT_SLOT,
                        tracker.slots[SILENCE_TIMEOUT_SLOT].initial_value,
                    )
                )
