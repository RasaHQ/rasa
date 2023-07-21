import abc
import copy
import json
import logging
import structlog
import re
from abc import ABC

import jsonpickle
import time
import uuid
from dateutil import parser
from datetime import datetime
from typing import (
    List,
    Dict,
    Text,
    Any,
    Type,
    Optional,
    TYPE_CHECKING,
    Iterable,
    cast,
    Tuple,
    TypeVar,
)

import rasa.shared.utils.common
import rasa.shared.utils.io
from typing import Union

from rasa.shared.constants import DOCS_URL_TRAINING_DATA
from rasa.shared.core.constants import (
    LOOP_NAME,
    EXTERNAL_MESSAGE_PREFIX,
    ACTION_NAME_SENDER_ID_CONNECTOR_STR,
    IS_EXTERNAL,
    USE_TEXT_FOR_FEATURIZATION,
    LOOP_INTERRUPTED,
    ENTITY_LABEL_SEPARATOR,
    ACTION_SESSION_START_NAME,
    ACTION_LISTEN_NAME,
)
from rasa.shared.exceptions import UnsupportedFeatureException
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_TYPE,
    INTENT,
    TEXT,
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ACTION_TEXT,
    ACTION_NAME,
    INTENT_NAME_KEY,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_GROUP,
    PREDICTED_CONFIDENCE_KEY,
    INTENT_RANKING_KEY,
    ENTITY_ATTRIBUTE_TEXT,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_CONFIDENCE,
    ENTITY_ATTRIBUTE_END,
    FULL_RETRIEVAL_INTENT_NAME_KEY,
)


if TYPE_CHECKING:
    from typing_extensions import TypedDict

    from rasa.shared.core.trackers import DialogueStateTracker

    EntityPrediction = TypedDict(
        "EntityPrediction",
        {
            ENTITY_ATTRIBUTE_TEXT: Text,  # type: ignore[misc]
            ENTITY_ATTRIBUTE_START: Optional[float],
            ENTITY_ATTRIBUTE_END: Optional[float],
            ENTITY_ATTRIBUTE_VALUE: Text,
            ENTITY_ATTRIBUTE_CONFIDENCE: float,
            ENTITY_ATTRIBUTE_TYPE: Text,
            ENTITY_ATTRIBUTE_GROUP: Optional[Text],
            ENTITY_ATTRIBUTE_ROLE: Optional[Text],
            "additional_info": Any,
        },
        total=False,
    )

    IntentPrediction = TypedDict(
        "IntentPrediction", {INTENT_NAME_KEY: Text, PREDICTED_CONFIDENCE_KEY: float}  # type: ignore[misc]  # noqa: E501
    )
    NLUPredictionData = TypedDict(
        "NLUPredictionData",
        {
            TEXT: Text,  # type: ignore[misc]
            INTENT: IntentPrediction,
            INTENT_RANKING_KEY: List[IntentPrediction],
            ENTITIES: List[EntityPrediction],
            "message_id": Optional[Text],
            "metadata": Dict,
        },
        total=False,
    )
logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


def deserialise_events(serialized_events: List[Dict[Text, Any]]) -> List["Event"]:
    """Convert a list of dictionaries to a list of corresponding events.

    Example format:
        [{"event": "slot", "value": 5, "name": "my_slot"}]
    """
    deserialised = []

    for e in serialized_events:
        if "event" in e:
            event = Event.from_parameters(e)
            if event:
                deserialised.append(event)
            else:
                structlogger.warning(
                    "event.deserialization.failed", rasa_event=copy.deepcopy(event)
                )

    return deserialised


def deserialise_entities(entities: Union[Text, List[Any]]) -> List[Dict[Text, Any]]:
    if isinstance(entities, str):
        entities = json.loads(entities)

    return [e for e in entities if isinstance(e, dict)]


def format_message(
    text: Text, intent: Optional[Text], entities: Union[Text, List[Any]]
) -> Text:
    """Uses NLU parser information to generate a message with inline entity annotations.

    Arguments:
        text: text of the message
        intent: intent of the message
        entities: entities of the message

    Return:
        Message with entities annotated inline, e.g.
        `I am from [Berlin]{`"`entity`"`: `"`city`"`}`.
    """
    from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataWriter
    from rasa.shared.nlu.training_data import entities_parser

    message_from_md = entities_parser.parse_training_example(text, intent)
    deserialised_entities = deserialise_entities(entities)
    return TrainingDataWriter.generate_message(
        {"text": message_from_md.get(TEXT), "entities": deserialised_entities}
    )


def split_events(
    events: Iterable["Event"],
    event_type_to_split_on: Type["Event"],
    additional_splitting_conditions: Optional[Dict[Text, Any]] = None,
    include_splitting_event: bool = True,
) -> List[List["Event"]]:
    """Splits events according to an event type and condition.

    Examples:
        Splitting events according to the event type `ActionExecuted` and the
        `action_name` 'action_session_start' would look as follows:

        >> _events = split_events(
                        events,
                        ActionExecuted,
                        {"action_name": "action_session_start"},
                        True
                     )

    Args:
        events: Events to split.
        event_type_to_split_on: The event type to split on.
        additional_splitting_conditions: Additional event attributes to split on.
        include_splitting_event: Whether the events of the type on which the split
            is based should be included in the returned events.

    Returns:
        The split events.
    """
    sub_events = []
    current: List["Event"] = []

    def event_fulfills_splitting_condition(evt: "Event") -> bool:
        # event does not have the correct type
        if not isinstance(evt, event_type_to_split_on):
            return False

        # the type is correct and there are no further conditions
        if not additional_splitting_conditions:
            return True

        # there are further conditions - check those
        return all(
            getattr(evt, k, None) == v
            for k, v in additional_splitting_conditions.items()
        )

    for event in events:
        if event_fulfills_splitting_condition(event):
            if current:
                sub_events.append(current)

            current = []
            if include_splitting_event:
                current.append(event)
        else:
            current.append(event)

    if current:
        sub_events.append(current)

    return sub_events


def do_events_begin_with_session_start(events: List["Event"]) -> bool:
    """Determines whether `events` begins with a session start sequence.

    A session start sequence is a sequence of two events: an executed
    `action_session_start` as well as a logged `session_started`.

    Args:
        events: The events to inspect.

    Returns:
        Whether `events` begins with a session start sequence.
    """
    if len(events) < 2:
        return False

    first = events[0]
    second = events[1]

    # We are not interested in specific metadata or timestamps. Action name and event
    # type are sufficient for this check
    return (
        isinstance(first, ActionExecuted)
        and first.action_name == ACTION_SESSION_START_NAME
        and isinstance(second, SessionStarted)
    )


def remove_parse_data(event: Dict[Text, Any]) -> Dict[Text, Any]:
    """Reduce event details to the minimum necessary to be structlogged.

    Deletes the parse_data key from the event if it exists.

    Args:
        event: The event to be reduced.

    Returns:
        A reduced copy of the event.
    """
    reduced_event = copy.deepcopy(event)
    if "parse_data" in reduced_event:
        del reduced_event["parse_data"]
    return reduced_event


E = TypeVar("E", bound="Event")


class Event(ABC):
    """Describes events in conversation and how the affect the conversation state.

    Immutable representation of everything which happened during a conversation of the
    user with the assistant. Tells the `rasa.shared.core.trackers.DialogueStateTracker`
    how to update its state as the events occur.
    """

    type_name = "event"

    def __init__(
        self,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def __ne__(self, other: Any) -> bool:
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)

    @abc.abstractmethod
    def as_story_string(self) -> Optional[Text]:
        """Returns the event as story string.

        Returns:
            textual representation of the event or None.
        """
        # Every class should implement this
        raise NotImplementedError

    @staticmethod
    def from_story_string(
        event_name: Text,
        parameters: Dict[Text, Any],
        default: Optional[Type["Event"]] = None,
    ) -> Optional[List["Event"]]:
        event_class = Event.resolve_by_type(event_name, default)

        if not event_class:
            return None

        return event_class._from_story_string(parameters)

    @staticmethod
    def from_parameters(
        parameters: Dict[Text, Any], default: Optional[Type["Event"]] = None
    ) -> Optional["Event"]:

        event_name = parameters.get("event")
        if event_name is None:
            return None

        event_class: Optional[Type[Event]] = Event.resolve_by_type(event_name, default)
        if not event_class:
            return None

        return event_class._from_parameters(parameters)

    @classmethod
    def _from_story_string(
        cls: Type[E], parameters: Dict[Text, Any]
    ) -> Optional[List[E]]:
        """Called to convert a parsed story line into an event."""
        return [cls(parameters.get("timestamp"), parameters.get("metadata"))]

    def as_dict(self) -> Dict[Text, Any]:
        d = {"event": self.type_name, "timestamp": self.timestamp}

        if self.metadata:
            d["metadata"] = self.metadata

        return d

    def fingerprint(self) -> Text:
        """Returns a unique hash for the event which is stable across python runs.

        Returns:
            fingerprint of the event
        """
        data = self.as_dict()
        del data["timestamp"]
        return rasa.shared.utils.io.get_dictionary_fingerprint(data)

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> Optional["Event"]:
        """Called to convert a dictionary of parameters to a single event.

        By default uses the same implementation as the story line
        conversation ``_from_story_string``. But the subclass might
        decide to handle parameters differently if the parsed parameters
        don't origin from a story file.
        """
        result = cls._from_story_string(parameters)
        if len(result) > 1:
            logger.warning(
                f"Event from parameters called with parameters "
                f"for multiple events. This is not supported, "
                f"only the first event will be returned. "
                f"Parameters: {parameters}"
            )
        return result[0] if result else None

    @staticmethod
    def resolve_by_type(
        type_name: Text, default: Optional[Type["Event"]] = None
    ) -> Optional[Type["Event"]]:
        """Returns a slots class by its type name."""
        for cls in rasa.shared.utils.common.all_subclasses(Event):
            if cls.type_name == type_name:
                return cls
        if type_name == "topic":
            return None  # backwards compatibility to support old TopicSet evts
        elif default is not None:
            return default
        else:
            raise ValueError(f"Unknown event name '{type_name}'.")

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state.

        Args:
            tracker: The current conversation state.
        """
        pass

    @abc.abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        # Every class should implement this
        raise NotImplementedError()

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return f"{self.__class__.__name__}()"


class AlwaysEqualEventMixin(Event, ABC):
    """Class to deduplicate common behavior for events without additional attributes."""

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, self.__class__):
            return NotImplemented

        return True


class SkipEventInMDStoryMixin(Event, ABC):
    """Skips the visualization of an event in Markdown stories."""

    def as_story_string(self) -> None:
        """Returns the event as story string.

        Returns:
            None, as this event should not appear inside the story.
        """
        return


class UserUttered(Event):
    """The user has said something to the bot.

    As a side effect a new `Turn` will be created in the `Tracker`.
    """

    type_name = "user"

    def __init__(
        self,
        text: Optional[Text] = None,
        intent: Optional[Dict] = None,
        entities: Optional[List[Dict]] = None,
        parse_data: Optional["NLUPredictionData"] = None,
        timestamp: Optional[float] = None,
        input_channel: Optional[Text] = None,
        message_id: Optional[Text] = None,
        metadata: Optional[Dict] = None,
        use_text_for_featurization: Optional[bool] = None,
    ) -> None:
        """Creates event for incoming user message.

        Args:
            text: Text of user message.
            intent: Intent prediction of user message.
            entities: Extracted entities.
            parse_data: Detailed NLU parsing result for message.
            timestamp: When the event was created.
            metadata: Additional event metadata.
            input_channel: Which channel the user used to send message.
            message_id: Unique ID for message.
            use_text_for_featurization: `True` if the message's text was used to predict
                next action. `False` if the message's intent was used.

        """
        self.text = text
        self.intent = intent if intent else {}
        self.entities = entities if entities else []
        self.input_channel = input_channel
        self.message_id = message_id

        super().__init__(timestamp, metadata)

        # The featurization is set by the policies during prediction time using a
        # `DefinePrevUserUtteredFeaturization` event.
        self.use_text_for_featurization = use_text_for_featurization
        # define how this user utterance should be featurized
        if self.text and not self.intent_name:
            # happens during training
            self.use_text_for_featurization = True
        elif self.intent_name and not self.text:
            # happens during training
            self.use_text_for_featurization = False

        self.parse_data: "NLUPredictionData" = {
            INTENT: self.intent,  # type: ignore[misc]
            # Copy entities so that changes to `self.entities` don't affect
            # `self.parse_data` and hence don't get persisted
            ENTITIES: self.entities.copy(),
            TEXT: self.text,
            "message_id": self.message_id,
            "metadata": self.metadata,
        }
        if parse_data:
            self.parse_data.update(**parse_data)

    @staticmethod
    def _from_parse_data(
        text: Text,
        parse_data: "NLUPredictionData",
        timestamp: Optional[float] = None,
        input_channel: Optional[Text] = None,
        message_id: Optional[Text] = None,
        metadata: Optional[Dict] = None,
    ) -> "UserUttered":
        return UserUttered(
            text,
            parse_data.get(INTENT),
            parse_data.get(ENTITIES, []),
            parse_data,
            timestamp,
            input_channel,
            message_id,
            metadata,
        )

    def __hash__(self) -> int:
        """Returns unique hash of object."""
        return hash(json.dumps(self.as_sub_state()))

    @property
    def intent_name(self) -> Optional[Text]:
        """Returns intent name or `None` if no intent."""
        return self.intent.get(INTENT_NAME_KEY)

    @property
    def full_retrieval_intent_name(self) -> Optional[Text]:
        """Returns full retrieval intent name or `None` if no retrieval intent."""
        return self.intent.get(FULL_RETRIEVAL_INTENT_NAME_KEY)

    # Note that this means two UserUttered events with the same text, intent
    # and entities but _different_ timestamps will be considered equal.
    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, UserUttered):
            return NotImplemented

        return (
            self.text,
            self.intent_name,
            [
                jsonpickle.encode(sorted(ent)) for ent in self.entities
            ],  # TODO: test? Or fix in regex_message_handler?
        ) == (
            other.text,
            other.intent_name,
            [jsonpickle.encode(sorted(ent)) for ent in other.entities],
        )

    def __str__(self) -> Text:
        """Returns text representation of event."""
        entities = ""
        if self.entities:
            entities_list = [
                f"{entity[ENTITY_ATTRIBUTE_VALUE]} "
                f"(Type: {entity[ENTITY_ATTRIBUTE_TYPE]}, "
                f"Role: {entity.get(ENTITY_ATTRIBUTE_ROLE)}, "
                f"Group: {entity.get(ENTITY_ATTRIBUTE_GROUP)})"
                for entity in self.entities
            ]
            entities = f", entities: {', '.join(entities_list)}"

        return (
            f"UserUttered(text: {self.text}, intent: {self.intent_name}"
            f"{entities}"
            f", use_text_for_featurization: {self.use_text_for_featurization})"
        )

    @staticmethod
    def empty() -> "UserUttered":
        return UserUttered(None)

    def is_empty(self) -> bool:
        return not self.text and not self.intent_name and not self.entities

    def as_dict(self) -> Dict[Text, Any]:
        _dict = super().as_dict()
        _dict.update(
            {
                "text": self.text,
                "parse_data": self.parse_data,
                "input_channel": getattr(self, "input_channel", None),
                "message_id": getattr(self, "message_id", None),
                "metadata": self.metadata,
            }
        )
        return _dict

    def as_sub_state(self) -> Dict[Text, Union[None, Text, List[Optional[Text]]]]:
        """Turns a UserUttered event into features.

        The substate contains information about entities, intent and text of the
        `UserUttered` event.

        Returns:
            a dictionary with intent name, text and entities
        """
        entities = [entity.get(ENTITY_ATTRIBUTE_TYPE) for entity in self.entities]
        entities.extend(
            (
                f"{entity.get(ENTITY_ATTRIBUTE_TYPE)}{ENTITY_LABEL_SEPARATOR}"
                f"{entity.get(ENTITY_ATTRIBUTE_ROLE)}"
            )
            for entity in self.entities
            if ENTITY_ATTRIBUTE_ROLE in entity
        )
        entities.extend(
            (
                f"{entity.get(ENTITY_ATTRIBUTE_TYPE)}{ENTITY_LABEL_SEPARATOR}"
                f"{entity.get(ENTITY_ATTRIBUTE_GROUP)}"
            )
            for entity in self.entities
            if ENTITY_ATTRIBUTE_GROUP in entity
        )

        out: Dict[Text, Union[None, Text, List[Optional[Text]]]] = {}
        # During training we expect either intent_name or text to be set.
        # During prediction both will be set.
        if self.text and (
            self.use_text_for_featurization or self.use_text_for_featurization is None
        ):
            out[TEXT] = self.text
        if self.intent_name and not self.use_text_for_featurization:
            out[INTENT] = self.intent_name
        # don't add entities for e2e utterances
        if entities and not self.use_text_for_featurization:
            out[ENTITIES] = entities

        return out

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["UserUttered"]]:
        try:
            return [
                cls._from_parse_data(
                    parameters.get("text"),
                    parameters.get("parse_data"),
                    parameters.get("timestamp"),
                    parameters.get("input_channel"),
                    parameters.get("message_id"),
                    parameters.get("metadata"),
                )
            ]
        except KeyError as e:
            raise ValueError(f"Failed to parse bot uttered event. {e}")

    def _entity_string(self) -> Text:
        if self.entities:
            return json.dumps(
                {
                    entity[ENTITY_ATTRIBUTE_TYPE]: entity[ENTITY_ATTRIBUTE_VALUE]
                    for entity in self.entities
                },
                ensure_ascii=False,
            )
        return ""

    def as_story_string(self, e2e: bool = False) -> Text:
        """Return event as string for Markdown training format.

        Args:
            e2e: `True` if the the event should be printed in the format for
                end-to-end conversation tests.

        Returns:
            Event as string.
        """
        if self.use_text_for_featurization and not e2e:
            raise UnsupportedFeatureException(
                f"Printing end-to-end user utterances is not supported in the "
                f"Markdown training format. Please use the YAML training data format "
                f"instead. Please see {DOCS_URL_TRAINING_DATA} for more information."
            )

        if e2e:
            text_with_entities = format_message(
                self.text or "", self.intent_name, self.entities
            )

            intent_prefix = f"{self.intent_name}: " if self.intent_name else ""
            return f"{intent_prefix}{text_with_entities}"

        return f"{self.intent_name or ''}{self._entity_string()}"

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to tracker. See docstring of `Event`."""
        tracker.latest_message = self
        tracker.clear_followup_action()

    @staticmethod
    def create_external(
        intent_name: Text,
        entity_list: Optional[List[Dict[Text, Any]]] = None,
        input_channel: Optional[Text] = None,
    ) -> "UserUttered":
        return UserUttered(
            text=f"{EXTERNAL_MESSAGE_PREFIX}{intent_name}",
            intent={INTENT_NAME_KEY: intent_name},
            metadata={IS_EXTERNAL: True},
            entities=entity_list or [],
            input_channel=input_channel,
        )


class DefinePrevUserUtteredFeaturization(SkipEventInMDStoryMixin):
    """Stores information whether action was predicted based on text or intent."""

    type_name = "user_featurization"

    def __init__(
        self,
        use_text_for_featurization: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event.

        Args:
            use_text_for_featurization: `True` if message text was used to predict
                action. `False` if intent was used.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        super().__init__(timestamp, metadata)
        self.use_text_for_featurization = use_text_for_featurization

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return f"DefinePrevUserUtteredFeaturization({self.use_text_for_featurization})"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.use_text_for_featurization)

    @classmethod
    def _from_parameters(
        cls, parameters: Dict[Text, Any]
    ) -> "DefinePrevUserUtteredFeaturization":
        return DefinePrevUserUtteredFeaturization(
            parameters.get(USE_TEXT_FOR_FEATURIZATION),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({USE_TEXT_FOR_FEATURIZATION: self.use_text_for_featurization})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state.

        Args:
            tracker: The current conversation state.
        """
        if tracker.latest_action_name != ACTION_LISTEN_NAME:
            # featurization belong only to the last user message
            # a user message is always followed by action listen
            return

        if not tracker.latest_message:
            return

        # update previous user message's featurization based on this event
        tracker.latest_message.use_text_for_featurization = (
            self.use_text_for_featurization
        )

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, DefinePrevUserUtteredFeaturization):
            return NotImplemented

        return self.use_text_for_featurization == other.use_text_for_featurization


class EntitiesAdded(SkipEventInMDStoryMixin):
    """Event that is used to add extracted entities to the tracker state."""

    type_name = "entities"

    def __init__(
        self,
        entities: List[Dict[Text, Any]],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Initializes event.

        Args:
            entities: Entities extracted from previous user message. This can either
                be done by NLU components or end-to-end policy predictions.
            timestamp: the timestamp
            metadata: some optional metadata
        """
        super().__init__(timestamp, metadata)
        self.entities = entities

    def __str__(self) -> Text:
        """Returns the string representation of the event."""
        entity_str = [e[ENTITY_ATTRIBUTE_TYPE] for e in self.entities]
        return f"{self.__class__.__name__}({entity_str})"

    def __hash__(self) -> int:
        """Returns the hash value of the event."""
        return hash(json.dumps(self.entities))

    def __eq__(self, other: Any) -> bool:
        """Compares this event with another event."""
        if not isinstance(other, EntitiesAdded):
            return NotImplemented

        return self.entities == other.entities

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "EntitiesAdded":
        return EntitiesAdded(
            parameters.get(ENTITIES),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Converts the event into a dict.

        Returns:
            A dict that represents this event.
        """
        d = super().as_dict()
        d.update({ENTITIES: self.entities})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state.

        Args:
            tracker: The current conversation state.
        """
        if tracker.latest_action_name != ACTION_LISTEN_NAME:
            # entities belong only to the last user message
            # a user message always comes after action listen
            return

        if not tracker.latest_message:
            return

        for entity in self.entities:
            if entity not in tracker.latest_message.entities:
                tracker.latest_message.entities.append(entity)


class BotUttered(SkipEventInMDStoryMixin):
    """The bot has said something to the user.

    This class is not used in the story training as it is contained in the

    ``ActionExecuted`` class. An entry is made in the ``Tracker``.
    """

    type_name = "bot"

    def __init__(
        self,
        text: Optional[Text] = None,
        data: Optional[Dict] = None,
        metadata: Optional[Dict[Text, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Creates event for a bot response.

        Args:
            text: Plain text which bot responded with.
            data: Additional data for more complex utterances (e.g. buttons).
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.text = text
        self.data = data or {}
        super().__init__(timestamp, metadata)

    def __members(self) -> Tuple[Optional[Text], Text, Text]:
        data_no_nones = {k: v for k, v in self.data.items() if v is not None}
        meta_no_nones = {k: v for k, v in self.metadata.items() if v is not None}
        return (
            self.text,
            jsonpickle.encode(data_no_nones),
            jsonpickle.encode(meta_no_nones),
        )

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.__members())

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, BotUttered):
            return NotImplemented

        return self.__members() == other.__members()

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return "BotUttered(text: {}, data: {}, metadata: {})".format(
            self.text, json.dumps(self.data), json.dumps(self.metadata)
        )

    def __repr__(self) -> Text:
        """Returns text representation of event for debugging."""
        return "BotUttered('{}', {}, {}, {})".format(
            self.text, json.dumps(self.data), json.dumps(self.metadata), self.timestamp
        )

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker.latest_bot_utterance = self

    def message(self) -> Dict[Text, Any]:
        """Return the complete message as a dictionary."""
        m = self.data.copy()
        m["text"] = self.text
        m["timestamp"] = self.timestamp
        m.update(self.metadata)

        if m.get("image") == m.get("attachment"):
            # we need this as there is an oddity we introduced a while ago where
            # we automatically set the attachment to the image. to not break
            # any persisted events we kept that, but we need to make sure that
            # the message contains the image only once
            m["attachment"] = None

        return m

    @staticmethod
    def empty() -> "BotUttered":
        """Creates an empty bot utterance."""
        return BotUttered()

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({"text": self.text, "data": self.data, "metadata": self.metadata})
        return d

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "BotUttered":
        try:
            return BotUttered(
                parameters.get("text"),
                parameters.get("data"),
                parameters.get("metadata"),
                parameters.get("timestamp"),
            )
        except KeyError as e:
            raise ValueError(f"Failed to parse bot uttered event. {e}")


class SlotSet(Event):
    """The user has specified their preference for the value of a `slot`.

    Every slot has a name and a value. This event can be used to set a
    value for a slot on a conversation.

    As a side effect the `Tracker`'s slots will be updated so
    that `tracker.slots[key]=value`.
    """

    type_name = "slot"

    def __init__(
        self,
        key: Text,
        value: Optional[Any] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event to set slot.

        Args:
            key: Name of the slot which is set.
            value: Value to which slot is set.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.key = key
        self.value = value
        super().__init__(timestamp, metadata)

    def __repr__(self) -> Text:
        """Returns text representation of event."""
        return f"SlotSet(key: {self.key}, value: {self.value})"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash((self.key, jsonpickle.encode(self.value)))

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, SlotSet):
            return NotImplemented

        return (self.key, self.value) == (other.key, other.value)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        props = json.dumps({self.key: self.value}, ensure_ascii=False)
        return f"{self.type_name}{props}"

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["SlotSet"]]:

        slots = []
        for slot_key, slot_val in parameters.items():
            slots.append(SlotSet(slot_key, slot_val))

        if slots:
            return slots
        else:
            return None

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({"name": self.key, "value": self.value})
        return d

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "SlotSet":
        try:
            return SlotSet(
                parameters.get("name"),
                parameters.get("value"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        except KeyError as e:
            raise ValueError(f"Failed to parse set slot event. {e}")

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker._set_slot(self.key, self.value)


class Restarted(AlwaysEqualEventMixin):
    """Conversation should start over & history wiped.

    Instead of deleting all events, this event can be used to reset the
    trackers state (e.g. ignoring any past user messages & resetting all
    the slots).
    """

    type_name = "restart"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124312)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Resets the tracker and triggers a followup `ActionSessionStart`."""
        tracker._reset()
        tracker.trigger_followup_action(ACTION_SESSION_START_NAME)


class UserUtteranceReverted(AlwaysEqualEventMixin):
    """Bot reverts everything until before the most recent user message.

    The bot will revert all events after the latest `UserUttered`, this
    also means that the last event on the tracker is usually `action_listen`
    and the bot is waiting for a new user message.
    """

    type_name = "rewind"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124315)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker._reset()
        tracker.replay_events()


class AllSlotsReset(AlwaysEqualEventMixin):
    """All Slots are reset to their initial values.

    If you want to keep the dialogue history and only want to reset the
    slots, you can use this event to set all the slots to their initial
    values.
    """

    type_name = "reset_slots"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124316)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker._reset_slots()


class ReminderScheduled(Event):
    """Schedules the asynchronous triggering of a user intent at a given time.

    The triggered intent can include entities if needed.
    """

    type_name = "reminder"

    def __init__(
        self,
        intent: Text,
        trigger_date_time: datetime,
        entities: Optional[List[Dict]] = None,
        name: Optional[Text] = None,
        kill_on_user_message: bool = True,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates the reminder.

        Args:
            intent: Name of the intent to be triggered.
            trigger_date_time: Date at which the execution of the action
                should be triggered (either utc or with tz).
            name: ID of the reminder. If there are multiple reminders with
                 the same id only the last will be run.
            entities: Entities that should be supplied together with the
                 triggered intent.
            kill_on_user_message: ``True`` means a user message before the
                 trigger date will abort the reminder.
            timestamp: Creation date of the event.
            metadata: Optional event metadata.
        """
        self.intent = intent
        self.entities = entities
        self.trigger_date_time = trigger_date_time
        self.kill_on_user_message = kill_on_user_message
        self.name = name if name is not None else str(uuid.uuid1())
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(
            (
                self.intent,
                self.entities,
                self.trigger_date_time.isoformat(),
                self.kill_on_user_message,
                self.name,
            )
        )

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, ReminderScheduled):
            return NotImplemented

        return self.name == other.name

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return (
            f"ReminderScheduled(intent: {self.intent}, "
            f"trigger_date: {self.trigger_date_time}, "
            f"entities: {self.entities}, name: {self.name})"
        )

    def scheduled_job_name(self, sender_id: Text) -> Text:
        return (
            f"[{hash(self.name)},{hash(self.intent)},{hash(str(self.entities))}]"
            f"{ACTION_NAME_SENDER_ID_CONNECTOR_STR}"
            f"{sender_id}"
        )

    def _properties(self) -> Dict[Text, Any]:
        return {
            "intent": self.intent,
            "date_time": self.trigger_date_time.isoformat(),
            "entities": self.entities,
            "name": self.name,
            "kill_on_user_msg": self.kill_on_user_message,
        }

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        props = json.dumps(self._properties())
        return f"{self.type_name}{props}"

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update(self._properties())
        return d

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["ReminderScheduled"]]:

        trigger_date_time = parser.parse(parameters.get("date_time"))

        return [
            ReminderScheduled(
                parameters.get("intent"),
                trigger_date_time,
                parameters.get("entities"),
                name=parameters.get("name"),
                kill_on_user_message=parameters.get("kill_on_user_msg", True),
                timestamp=parameters.get("timestamp"),
                metadata=parameters.get("metadata"),
            )
        ]


class ReminderCancelled(Event):
    """Cancel certain jobs."""

    type_name = "cancel_reminder"

    def __init__(
        self,
        name: Optional[Text] = None,
        intent: Optional[Text] = None,
        entities: Optional[List[Dict]] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates a ReminderCancelled event.

        If all arguments are `None`, this will cancel all reminders.
        are to be cancelled. If no arguments are supplied, this will cancel all
        reminders.

        Args:
            name: Name of the reminder to be cancelled.
            intent: Intent name that is to be used to identify the reminders to be
                cancelled.
            entities: Entities that are to be used to identify the reminders to be
                cancelled.
            timestamp: Optional timestamp.
            metadata: Optional event metadata.
        """
        self.name = name
        self.intent = intent
        self.entities = entities
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash((self.name, self.intent, str(self.entities)))

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, ReminderCancelled):
            return NotImplemented

        return hash(self) == hash(other)

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return (
            f"ReminderCancelled(name: {self.name}, intent: {self.intent}, "
            f"entities: {self.entities})"
        )

    def cancels_job_with_name(self, job_name: Text, sender_id: Text) -> bool:
        """Determines if this event should cancel the job with the given name.

        Args:
            job_name: Name of the job to be tested.
            sender_id: The `sender_id` of the tracker.

        Returns:
            `True`, if this `ReminderCancelled` event should cancel the job with the
            given name, and `False` otherwise.
        """
        match = re.match(
            rf"^\[([\d\-]*),([\d\-]*),([\d\-]*)\]"
            rf"({re.escape(ACTION_NAME_SENDER_ID_CONNECTOR_STR)}"
            rf"{re.escape(sender_id)})",
            job_name,
        )
        if not match:
            return False
        name_hash, intent_hash, entities_hash = match.group(1, 2, 3)

        # Cancel everything unless names/intents/entities are given to
        # narrow it down.
        return (
            ((not self.name) or self._matches_name_hash(name_hash))
            and ((not self.intent) or self._matches_intent_hash(intent_hash))
            and ((not self.entities) or self._matches_entities_hash(entities_hash))
        )

    def _matches_name_hash(self, name_hash: Text) -> bool:
        return str(hash(self.name)) == name_hash

    def _matches_intent_hash(self, intent_hash: Text) -> bool:
        return str(hash(self.intent)) == intent_hash

    def _matches_entities_hash(self, entities_hash: Text) -> bool:
        return str(hash(str(self.entities))) == entities_hash

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        props = json.dumps(
            {"name": self.name, "intent": self.intent, "entities": self.entities}
        )
        return f"{self.type_name}{props}"

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["ReminderCancelled"]]:
        return [
            ReminderCancelled(
                parameters.get("name"),
                parameters.get("intent"),
                parameters.get("entities"),
                timestamp=parameters.get("timestamp"),
                metadata=parameters.get("metadata"),
            )
        ]


class ActionReverted(AlwaysEqualEventMixin):
    """Bot undoes its last action.

    The bot reverts everything until before the most recent action.
    This includes the action itself, as well as any events that
    action created, like set slot events - the bot will now
    predict a new action using the state before the most recent
    action.
    """

    type_name = "undo"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124318)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker._reset()
        tracker.replay_events()


class StoryExported(Event):
    """Story should get dumped to a file."""

    type_name = "export"

    def __init__(
        self,
        path: Optional[Text] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event about story exporting.

        Args:
            path: Path to which story was exported to.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.path = path
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124319)

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["StoryExported"]]:
        return [
            StoryExported(
                parameters.get("path"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        ]

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        if self.path:
            tracker.export_stories_to_file(self.path)

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, StoryExported):
            return NotImplemented

        return self.path == other.path


class FollowupAction(Event):
    """Enqueue a followup action."""

    type_name = "followup"

    def __init__(
        self,
        name: Text,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates an event which forces the model to run a certain action next.

        Args:
            name: Name of the action to run.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.action_name = name
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.action_name)

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, FollowupAction):
            return NotImplemented

        return self.action_name == other.action_name

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return f"FollowupAction(action: {self.action_name})"

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        props = json.dumps({"name": self.action_name})
        return f"{self.type_name}{props}"

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["FollowupAction"]]:

        return [
            FollowupAction(
                parameters.get("name"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        ]

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({"name": self.action_name})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker.trigger_followup_action(self.action_name)


class ConversationPaused(AlwaysEqualEventMixin):
    """Ignore messages from the user to let a human take over.

    As a side effect the `Tracker`'s `paused` attribute will
    be set to `True`.
    """

    type_name = "pause"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124313)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return str(self)

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker._paused = True


class ConversationResumed(AlwaysEqualEventMixin):
    """Bot takes over conversation.

    Inverse of `PauseConversation`. As a side effect the `Tracker`'s
    `paused` attribute will be set to `False`.
    """

    type_name = "resume"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124314)

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        return self.type_name

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker._paused = False


class ActionExecuted(Event):
    """An operation describes an action taken + its result.

    It comprises an action and a list of events. operations will be appended
    to the latest `Turn`` in `Tracker.turns`.
    """

    type_name = "action"

    def __init__(
        self,
        action_name: Optional[Text] = None,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
        action_text: Optional[Text] = None,
        hide_rule_turn: bool = False,
    ) -> None:
        """Creates event for a successful event execution.

        Args:
            action_name: Name of the action which was executed. `None` if it was an
                end-to-end prediction.
            policy: Policy which predicted action.
            confidence: Confidence with which policy predicted action.
            timestamp: When the event was created.
            metadata: Additional event metadata.
            action_text: In case it's an end-to-end action prediction, the text which
                was predicted.
            hide_rule_turn: If `True`, this action should be hidden in the dialogue
                history created for ML-based policies.
        """
        self.action_name = action_name
        self.policy = policy
        self.confidence = confidence
        self.unpredictable = False
        self.action_text = action_text
        self.hide_rule_turn = hide_rule_turn

        if self.action_name is None and self.action_text is None:
            raise ValueError(
                "Both the name of the action and the end-to-end "
                "predicted text are missing. "
                "The `ActionExecuted` event cannot be initialised."
            )

        super().__init__(timestamp, metadata)

    def __members__(self) -> Tuple[Optional[Text], Optional[Text], Text]:
        meta_no_nones = {k: v for k, v in self.metadata.items() if v is not None}
        return (self.action_name, self.action_text, jsonpickle.encode(meta_no_nones))

    def __repr__(self) -> Text:
        """Returns event as string for debugging."""
        return "ActionExecuted(action: {}, policy: {}, confidence: {})".format(
            self.action_name, self.policy, self.confidence
        )

    def __str__(self) -> Text:
        """Returns event as human readable string."""
        return str(self.action_name) or str(self.action_text)

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.__members__())

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, ActionExecuted):
            return NotImplemented

        return self.__members__() == other.__members__()

    def as_story_string(self) -> Optional[Text]:
        """Returns event in Markdown format."""
        if self.action_text:
            raise UnsupportedFeatureException(
                f"Printing end-to-end bot utterances is not supported in the "
                f"Markdown training format. Please use the YAML training data format "
                f"instead. Please see {DOCS_URL_TRAINING_DATA} for more information."
            )

        return self.action_name

    @classmethod
    def _from_story_string(
        cls, parameters: Dict[Text, Any]
    ) -> Optional[List["ActionExecuted"]]:
        return [
            ActionExecuted(
                parameters.get("name"),
                parameters.get("policy"),
                parameters.get("confidence"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
                parameters.get("action_text"),
                parameters.get("hide_rule_turn", False),
            )
        ]

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update(
            {
                "name": self.action_name,
                "policy": self.policy,
                "confidence": self.confidence,
                "action_text": self.action_text,
                "hide_rule_turn": self.hide_rule_turn,
            }
        )
        return d

    def as_sub_state(self) -> Dict[Text, Text]:
        """Turns ActionExecuted into a dictionary containing action name or action text.

        One action cannot have both set at the same time

        Returns:
            a dictionary containing action name or action text with the corresponding
            key.
        """
        if self.action_name:
            return {ACTION_NAME: self.action_name}
        else:
            # FIXME: we should define the type better here, and require either
            #        `action_name` or `action_text`
            return {ACTION_TEXT: cast(Text, self.action_text)}

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker.set_latest_action(self.as_sub_state())
        tracker.clear_followup_action()


class AgentUttered(SkipEventInMDStoryMixin):
    """The agent has said something to the user.

    This class is not used in the story training as it is contained in the
    ``ActionExecuted`` class. An entry is made in the ``Tracker``.
    """

    type_name = "agent"

    def __init__(
        self,
        text: Optional[Text] = None,
        data: Optional[Any] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """See docstring of `BotUttered`."""
        self.text = text
        self.data = data
        super().__init__(timestamp, metadata)

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash((self.text, jsonpickle.encode(self.data)))

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, AgentUttered):
            return NotImplemented

        return (self.text, jsonpickle.encode(self.data)) == (
            other.text,
            jsonpickle.encode(other.data),
        )

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return "AgentUttered(text: {}, data: {})".format(
            self.text, json.dumps(self.data)
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({"text": self.text, "data": self.data})
        return d

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "AgentUttered":
        try:
            return AgentUttered(
                parameters.get("text"),
                parameters.get("data"),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        except KeyError as e:
            raise ValueError(f"Failed to parse agent uttered event. {e}")


class ActiveLoop(Event):
    """If `name` is given: activates a loop with `name` else deactivates active loop."""

    type_name = "active_loop"

    def __init__(
        self,
        name: Optional[Text],
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event for active loop.

        Args:
            name: Name of activated loop or `None` if current loop is deactivated.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.name = name
        super().__init__(timestamp, metadata)

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return f"Loop({self.name})"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, ActiveLoop):
            return NotImplemented

        return self.name == other.name

    def as_story_string(self) -> Text:
        """Returns text representation of event."""
        props = json.dumps({LOOP_NAME: self.name})
        return f"{ActiveLoop.type_name}{props}"

    @classmethod
    def _from_story_string(cls, parameters: Dict[Text, Any]) -> List["ActiveLoop"]:
        """Called to convert a parsed story line into an event."""
        return [
            ActiveLoop(
                parameters.get(LOOP_NAME),
                parameters.get("timestamp"),
                parameters.get("metadata"),
            )
        ]

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({LOOP_NAME: self.name})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker.change_loop_to(self.name)


class LegacyForm(ActiveLoop):
    """Legacy handler of old `Form` events.

    The `ActiveLoop` event used to be called `Form`. This class is there to handle old
    legacy events which were stored with the old type name `form`.
    """

    type_name = "form"

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        # Dump old `Form` events as `ActiveLoop` events instead of keeping the old
        # event type.
        d["event"] = ActiveLoop.type_name
        return d

    def fingerprint(self) -> Text:
        """Returns the hash of the event."""
        d = self.as_dict()
        # Revert event name to legacy subclass name to avoid different event types
        # having the same fingerprint.
        d["event"] = self.type_name
        del d["timestamp"]
        return rasa.shared.utils.io.get_dictionary_fingerprint(d)


class LoopInterrupted(SkipEventInMDStoryMixin):
    """Event added by FormPolicy and RulePolicy.

    Notifies form action whether or not to validate the user input.
    """

    type_name = "loop_interrupted"

    def __init__(
        self,
        is_interrupted: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Event to notify that loop was interrupted.

        This e.g. happens when a user is within a form, and is de-railing the
        form-filling by asking FAQs.

        Args:
            is_interrupted: `True` if the loop execution was interrupted, and ML
                policies had to take over the last prediction.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        super().__init__(timestamp, metadata)
        self.is_interrupted = is_interrupted

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return f"{LoopInterrupted.__name__}({self.is_interrupted})"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.is_interrupted)

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, LoopInterrupted):
            return NotImplemented

        return self.is_interrupted == other.is_interrupted

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "LoopInterrupted":
        return LoopInterrupted(
            parameters.get(LOOP_INTERRUPTED, False),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update({LOOP_INTERRUPTED: self.is_interrupted})
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker.interrupt_loop(self.is_interrupted)


class LegacyFormValidation(LoopInterrupted):
    """Legacy handler of old `FormValidation` events.

    The `LoopInterrupted` event used to be called `FormValidation`. This class is there
    to handle old legacy events which were stored with the old type name
    `form_validation`.
    """

    type_name = "form_validation"

    def __init__(
        self,
        validate: bool,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """See parent class docstring."""
        # `validate = True` is the same as `interrupted = False`
        super().__init__(not validate, timestamp, metadata)

    @classmethod
    def _from_parameters(cls, parameters: Dict) -> "LoopInterrupted":
        return LoopInterrupted(
            # `validate = True` means `is_interrupted = False`
            not parameters.get("validate", True),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        # Dump old `Form` events as `ActiveLoop` events instead of keeping the old
        # event type.
        d["event"] = LoopInterrupted.type_name
        return d

    def fingerprint(self) -> Text:
        """Returns hash of the event."""
        d = self.as_dict()
        # Revert event name to legacy subclass name to avoid different event types
        # having the same fingerprint.
        d["event"] = self.type_name
        del d["timestamp"]
        return rasa.shared.utils.io.get_dictionary_fingerprint(d)


class ActionExecutionRejected(SkipEventInMDStoryMixin):
    """Notify Core that the execution of the action has been rejected."""

    type_name = "action_execution_rejected"

    def __init__(
        self,
        action_name: Text,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates event.

        Args:
            action_name: Action which was rejected.
            policy: Policy which predicted the rejected action.
            confidence: Confidence with which the reject action was predicted.
            timestamp: When the event was created.
            metadata: Additional event metadata.
        """
        self.action_name = action_name
        self.policy = policy
        self.confidence = confidence
        super().__init__(timestamp, metadata)

    def __str__(self) -> Text:
        """Returns text representation of event."""
        return (
            "ActionExecutionRejected("
            "action: {}, policy: {}, confidence: {})"
            "".format(self.action_name, self.policy, self.confidence)
        )

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(self.action_name)

    def __eq__(self, other: Any) -> bool:
        """Compares object with other object."""
        if not isinstance(other, ActionExecutionRejected):
            return NotImplemented

        return self.action_name == other.action_name

    @classmethod
    def _from_parameters(cls, parameters: Dict[Text, Any]) -> "ActionExecutionRejected":
        return ActionExecutionRejected(
            parameters.get("name"),
            parameters.get("policy"),
            parameters.get("confidence"),
            parameters.get("timestamp"),
            parameters.get("metadata"),
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns serialized event."""
        d = super().as_dict()
        d.update(
            {
                "name": self.action_name,
                "policy": self.policy,
                "confidence": self.confidence,
            }
        )
        return d

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        tracker.reject_action(self.action_name)


class SessionStarted(AlwaysEqualEventMixin):
    """Mark the beginning of a new conversation session."""

    type_name = "session_started"

    def __hash__(self) -> int:
        """Returns unique hash for event."""
        return hash(32143124320)

    def as_story_string(self) -> None:
        """Skips representing event in stories."""
        logger.warning(
            f"'{self.type_name}' events cannot be serialised as story strings."
        )

    def apply_to(self, tracker: "DialogueStateTracker") -> None:
        """Applies event to current conversation state."""
        # noinspection PyProtectedMember
        tracker._reset()
