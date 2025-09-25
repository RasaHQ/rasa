from __future__ import annotations

import json
from inspect import isawaitable, iscoroutinefunction
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Text,
    TypeVar,
    Union,
)

import structlog

import rasa.shared.utils.cli
import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.utils.json_utils
from rasa.core.brokers.broker import EventBroker
from rasa.plugin import plugin_manager
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.conversation import Dialogue
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import (
    ActionExecuted,
    DialogueStateTracker,
    TrackerEventDiffEngine,
)
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.get_logger(__name__)


def check_if_tracker_store_async(tracker_store: TrackerStore) -> bool:
    """Evaluates if a tracker store object is async based on implementation of methods.

    :param tracker_store: tracker store object we're evaluating
    :return: if the tracker store correctly implements all async methods
    """
    return all(
        iscoroutinefunction(getattr(tracker_store, method))
        for method in _get_async_tracker_store_methods()
    )


def _get_async_tracker_store_methods() -> List[str]:
    return [
        attribute
        for attribute in dir(TrackerStore)
        if iscoroutinefunction(getattr(TrackerStore, attribute))
    ]


class TrackerDeserialisationException(RasaException):
    """Raised when an error is encountered while deserialising a tracker."""


SerializationType = TypeVar("SerializationType")


class SerializedTrackerRepresentation(Generic[SerializationType]):
    """Mixin class for specifying different serialization methods per tracker store."""

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> SerializationType:
        """Requires implementation to return representation of tracker."""
        raise NotImplementedError()


class SerializedTrackerAsText(SerializedTrackerRepresentation[Text]):
    """Mixin class that returns the serialized tracker as string."""

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> Text:
        """Serializes the tracker, returns representation of the tracker."""
        dialogue = tracker.as_dialogue()

        return json.dumps(dialogue.as_dict())


class SerializedTrackerAsDict(SerializedTrackerRepresentation[Dict]):
    """Mixin class that returns the serialized tracker as dictionary."""

    @staticmethod
    def serialise_tracker(tracker: DialogueStateTracker) -> Dict:
        """Serializes the tracker, returns representation of the tracker."""
        d = tracker.as_dialogue().as_dict()
        d.update({"sender_id": tracker.sender_id})
        return d


class TrackerStore:
    """Represents common behavior and interface for all `TrackerStore`s."""

    def __init__(
        self,
        domain: Optional[Domain],
        event_broker: Optional[EventBroker] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Create a TrackerStore.

        Args:
            domain: The `Domain` to initialize the `DialogueStateTracker`.
            event_broker: An event broker to publish any new events to another
                destination.
            kwargs: Additional kwargs.
        """
        self._domain = domain or Domain.empty()
        self.event_broker = event_broker
        self.max_event_history: Optional[int] = None

    @staticmethod
    def create(
        obj: Union[TrackerStore, EndpointConfig, None],
        domain: Optional[Domain] = None,
        event_broker: Optional[EventBroker] = None,
    ) -> TrackerStore:
        """Factory to create a tracker store."""
        if isinstance(obj, TrackerStore):
            return obj

        import pymongo.errors
        import sqlalchemy.exc
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            _tracker_store = plugin_manager().hook.create_tracker_store(
                endpoint_config=obj,
                domain=domain,
                event_broker=event_broker,
            )

            tracker_store = (
                _tracker_store
                if _tracker_store
                else create_tracker_store(obj, domain, event_broker)
            )

            return tracker_store
        except (
            BotoCoreError,
            ClientError,
            pymongo.errors.ConnectionFailure,
            sqlalchemy.exc.OperationalError,
            ConnectionError,
            pymongo.errors.OperationFailure,
        ) as error:
            raise ConnectionException(
                "Cannot connect to tracker store." + str(error)
            ) from error

    async def get_or_create_tracker(
        self,
        sender_id: Text,
        max_event_history: Optional[int] = None,
        append_action_listen: bool = True,
    ) -> "DialogueStateTracker":
        """Returns tracker or creates one if the retrieval returns None.

        Args:
            sender_id: Conversation ID associated with the requested tracker.
            max_event_history: Value to update the tracker store's max event history to.
            append_action_listen: Whether or not to append an initial `action_listen`.
        """
        self.max_event_history = max_event_history

        tracker = await self.retrieve(sender_id)

        if tracker is None:
            tracker = await self.create_tracker(
                sender_id, append_action_listen=append_action_listen
            )

        return tracker

    def init_tracker(self, sender_id: Text) -> "DialogueStateTracker":
        """Returns a Dialogue State Tracker."""
        return DialogueStateTracker(
            sender_id,
            self.domain.slots,
            max_event_history=self.max_event_history,
        )

    async def create_tracker(
        self, sender_id: Text, append_action_listen: bool = True
    ) -> DialogueStateTracker:
        """Creates a new tracker for `sender_id`.

        The tracker begins with a `SessionStarted` event and is initially listening.

        Args:
            sender_id: Conversation ID associated with the tracker.
            append_action_listen: Whether or not to append an initial `action_listen`.

        Returns:
            The newly created tracker for `sender_id`.
        """
        tracker = self.init_tracker(sender_id)

        if append_action_listen:
            tracker.update(ActionExecuted(ACTION_LISTEN_NAME))

        await self.save(tracker)

        return tracker

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Save method that will be overridden by specific tracker."""
        raise NotImplementedError()

    async def exists(self, conversation_id: Text) -> bool:
        """Checks if tracker exists for the specified ID.

        This method may be overridden by the specific tracker store for
        faster implementations.

        Args:
            conversation_id: Conversation ID to check if the tracker exists.

        Returns:
            `True` if the tracker exists, `False` otherwise.
        """
        return await self.retrieve(conversation_id) is not None

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Retrieves tracker for the latest conversation session.

        This method will be overridden by the specific tracker store.

        Args:
            sender_id: Conversation ID to fetch the tracker for.

        Returns:
            Tracker containing events from the latest conversation sessions.
        """
        raise NotImplementedError()

    async def delete(self, sender_id: str) -> None:
        """Delete tracker for the given sender_id.

        Args:
            sender_id: Conversation ID to delete the tracker for.
        """
        raise NotImplementedError()

    async def update(self, tracker: DialogueStateTracker) -> None:
        """Replace an existing tracker with a new one.

        Args:
            tracker: The tracker to update.
        """
        raise NotImplementedError()

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Retrieve method for fetching all tracker events.

        Fetches events across conversation sessions. The default implementation
        uses `self.retrieve()`.

        Args:
            conversation_id: The conversation ID to retrieve the tracker for.

        Returns:
            The fetch tracker containing all events across session starts.
        """
        return await self.retrieve(conversation_id)

    async def get_or_create_full_tracker(
        self,
        sender_id: Text,
        append_action_listen: bool = True,
    ) -> "DialogueStateTracker":
        """Returns tracker or creates one if the retrieval returns None.

        Args:
            sender_id: Conversation ID associated with the requested tracker.
            append_action_listen: Whether to append an initial `action_listen`.

        Returns:
            The tracker for the conversation ID.
        """
        tracker = await self.retrieve_full_tracker(sender_id)

        if tracker is None:
            tracker = await self.create_tracker(
                sender_id, append_action_listen=append_action_listen
            )

        return tracker

    async def stream_events(self, tracker: DialogueStateTracker) -> None:
        """Streams events to a message broker."""
        if self.event_broker is None:
            structlogger.debug(
                "tracker_store.stream_events.no_broker_configured",
                event_info="No event broker configured. Skipping streaming events.",
            )
            return None

        if (
            hasattr(self.event_broker, "stream_pii")
            and not self.event_broker.stream_pii
        ):
            # If the event broker is configured to not stream un-anonymized events,
            # skip streaming
            structlogger.debug(
                "tracker_store.stream_events.no_streaming",
                event_info="Un-anonymized events will not be published "
                "to the event broker.",
            )
            return None

        old_tracker = await self.retrieve(tracker.sender_id)
        new_events = TrackerEventDiffEngine.event_difference(old_tracker, tracker)

        await self._stream_new_events(self.event_broker, new_events, tracker.sender_id)

    async def _stream_new_events(
        self,
        event_broker: EventBroker,
        new_events: List[Event],
        sender_id: Text,
    ) -> None:
        """Publishes new tracker events to a message broker."""
        for event in new_events:
            body = {"sender_id": sender_id}
            body.update(event.as_dict())
            event_broker.publish(body)

    async def keys(self) -> Iterable[Text]:
        """Returns the set of values for the tracker store's primary key."""
        raise NotImplementedError()

    async def count_conversations(self, after_timestamp: float = 0.0) -> int:
        """Returns the number of conversations that have occurred after a timestamp.

        By default, this method returns the number of conversations that
        have occurred after the Unix epoch (i.e. timestamp 0). A conversation
        is considered to have occurred after a timestamp if at least one event
        happened after that timestamp.
        """
        tracker_keys = await self.keys()

        conversation_count = 0
        for key in tracker_keys:
            tracker = await self.retrieve(key)
            if tracker is None or not tracker.events:
                continue

            last_event = tracker.events[-1]
            if last_event.timestamp >= after_timestamp:
                conversation_count += 1

        return conversation_count

    def deserialise_tracker(
        self, sender_id: Text, serialised_tracker: Union[Text, bytes]
    ) -> Optional[DialogueStateTracker]:
        """Deserializes the tracker and returns it."""
        tracker = self.init_tracker(sender_id)

        try:
            dialogue = Dialogue.from_parameters(json.loads(serialised_tracker))
        except UnicodeDecodeError as e:
            raise TrackerDeserialisationException(
                "Tracker cannot be deserialised. "
                "Trackers must be serialised as json. "
                "Support for deserialising pickled trackers has been removed."
            ) from e

        tracker.recreate_from_dialogue(dialogue)

        return tracker

    @property
    def domain(self) -> Domain:
        """Returns the domain of the tracker store."""
        return self._domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        self._domain = domain or Domain.empty()


class InMemoryTrackerStore(TrackerStore, SerializedTrackerAsText):
    """Stores conversation history in memory."""

    def __init__(
        self,
        domain: Domain,
        event_broker: Optional[EventBroker] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Initializes the tracker store."""
        self.store: Dict[Text, Text] = {}
        super().__init__(domain, event_broker, **kwargs)

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Updates and saves the current conversation state."""
        await self.stream_events(tracker)
        serialised = InMemoryTrackerStore.serialise_tracker(tracker)
        self.store[tracker.sender_id] = serialised

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id."""
        if sender_id not in self.store:
            structlogger.info(
                "in_memory_tracker_store.delete.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        del self.store[sender_id]

        structlogger.info(
            "in_memory_tracker_store.delete.deleted_tracker",
            sender_id=sender_id,
        )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id."""
        return await self._retrieve(sender_id, fetch_all_sessions=False)

    async def keys(self) -> Iterable[Text]:
        """Returns sender_ids of the Tracker Store in memory."""
        return self.store.keys()

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
        """
        return await self._retrieve(sender_id, fetch_all_sessions=True)

    async def _retrieve(
        self, sender_id: Text, fetch_all_sessions: bool
    ) -> Optional[DialogueStateTracker]:
        """Returns tracker matching sender_id.

        Args:
            sender_id: Conversation ID to fetch the tracker for.
            fetch_all_sessions: Whether to fetch all sessions or only the last one.
        """
        if sender_id not in self.store:
            structlogger.debug(
                "in_memory_tracker_store.retrieve.no_tracker_for_sender_id",
                event_info=f"Could not find tracker for conversation ID '{sender_id}'.",
            )
            return None

        tracker = self.deserialise_tracker(sender_id, self.store[sender_id])

        if not tracker:
            structlogger.debug(
                "in_memory_tracker_store.retrieve.failed_to_deserialize_tracker",
                event_info=(
                    f"Could not deserialize tracker "
                    f"for conversation ID '{sender_id}'.",
                ),
            )
            return None

        if fetch_all_sessions:
            return tracker

        # only return the last session
        multiple_tracker_sessions = (
            rasa.shared.core.trackers.get_trackers_for_conversation_sessions(tracker)
        )

        if len(multiple_tracker_sessions) <= 1:
            return tracker

        return multiple_tracker_sessions[-1]

    async def update(self, tracker: DialogueStateTracker) -> None:
        """Replace an existing tracker with a new one.

        Args:
            tracker: The tracker to update.
        """
        await self.save(tracker)


def validate_port(port: Any) -> Optional[int]:
    """Ensure that port can be converted to integer.

    Raises:
        RasaException if port cannot be cast to integer.
    """
    if port is not None and not isinstance(port, int):
        try:
            port = int(port)
        except ValueError as e:
            raise RasaException(f"The port '{port}' cannot be cast to integer.") from e

    return port


class FailSafeTrackerStore(TrackerStore):
    """Tracker store wrapper.

    Allows a fallback to a different tracker store in case of errors.
    """

    def __init__(
        self,
        tracker_store: TrackerStore,
        on_tracker_store_error: Optional[Callable[[Exception], None]] = None,
        fallback_tracker_store: Optional[TrackerStore] = None,
    ) -> None:
        """Create a `FailSafeTrackerStore`.

        Args:
            tracker_store: Primary tracker store.
            on_tracker_store_error: Callback which is called when there is an error
                in the primary tracker store.
            fallback_tracker_store: Fallback tracker store.
        """
        self._fallback_tracker_store: Optional[TrackerStore] = fallback_tracker_store
        self._tracker_store = tracker_store
        self._on_tracker_store_error = on_tracker_store_error

        super().__init__(tracker_store.domain, tracker_store.event_broker)

    @property
    def domain(self) -> Domain:
        """Returns the domain of the primary tracker store."""
        return self._tracker_store.domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        self._tracker_store.domain = domain

        if self._fallback_tracker_store:
            self._fallback_tracker_store.domain = domain

    @property
    def fallback_tracker_store(self) -> TrackerStore:
        """Returns the fallback tracker store."""
        if not self._fallback_tracker_store:
            self._fallback_tracker_store = InMemoryTrackerStore(
                self._tracker_store.domain, self._tracker_store.event_broker
            )

        return self._fallback_tracker_store

    def on_tracker_store_error(self, error: Exception) -> None:
        """Calls the callback when there is an error in the primary tracker store."""
        if self._on_tracker_store_error:
            self._on_tracker_store_error(error)
        else:
            structlogger.error(
                "fail_safe_tracker_store.tracker_store_error",
                event_info=(
                    f"Error happened when trying to save conversation tracker to "
                    f"'{self._tracker_store.__class__.__name__}'. Falling back to use "
                    f"the '{InMemoryTrackerStore.__name__}'. Please "
                    f"investigate the following error: {error}."
                ),
            )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Calls `retrieve` method of primary tracker store."""
        try:
            return await self._tracker_store.retrieve(sender_id)
        except Exception as e:
            self.on_tracker_store_retrieve_error(e)
            return None

    async def keys(self) -> Iterable[Text]:
        """Calls `keys` method of primary tracker store."""
        try:
            return await self._tracker_store.keys()
        except Exception as e:
            self.on_tracker_store_error(e)
            return []

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Calls `save` method of primary tracker store."""
        try:
            await self._tracker_store.save(tracker)
        except Exception as e:
            self.on_tracker_store_error(e)
            await self.fallback_tracker_store.save(tracker)

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id."""
        try:
            await self._tracker_store.delete(sender_id)
        except Exception as e:
            self.on_tracker_store_error(e)
            await self.fallback_tracker_store.delete(sender_id)

    async def update(self, tracker: DialogueStateTracker) -> None:
        """Replace an existing tracker with a new one."""
        try:
            await self._tracker_store.update(tracker)
        except Exception as e:
            self.on_tracker_store_error(e)
            await self.fallback_tracker_store.update(tracker)

    async def retrieve_full_tracker(
        self, sender_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Calls `retrieve_full_tracker` method of primary tracker store.

        Args:
            sender_id: The sender id of the tracker to retrieve.
        """
        try:
            return await self._tracker_store.retrieve_full_tracker(sender_id)
        except Exception as e:
            self.on_tracker_store_retrieve_error(e)
            return None

    def on_tracker_store_retrieve_error(self, error: Exception) -> None:
        """Calls `_on_tracker_store_error` callable attribute if set.

        Otherwise, logs the error.

        Args:
            error: The error that occurred.
        """
        if self._on_tracker_store_error:
            self._on_tracker_store_error(error)
        else:
            structlogger.error(
                "fail_safe_tracker_store.tracker_store_retrieve_error",
                event_info=(
                    f"Error happened when trying to retrieve conversation tracker from "
                    f"'{self._tracker_store.__class__.__name__}'. Falling back to use "
                    f"the '{InMemoryTrackerStore.__name__}'."
                ),
                exec_info=error,
            )


def _create_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
    domain: Optional[Domain] = None,
    event_broker: Optional[EventBroker] = None,
) -> TrackerStore:
    """Given an endpoint configuration, create a proper tracker store object."""
    from rasa.core.tracker_stores.dynamo_tracker_store import DynamoTrackerStore
    from rasa.core.tracker_stores.mongo_tracker_store import MongoTrackerStore
    from rasa.core.tracker_stores.redis_tracker_store import (
        RedisTrackerStore,
    )
    from rasa.core.tracker_stores.sql_tracker_store import SQLTrackerStore

    domain = domain or Domain.empty()

    if endpoint_config is None or endpoint_config.type is None:
        # default tracker store if no type is set
        tracker_store: TrackerStore = InMemoryTrackerStore(domain, event_broker)
    elif endpoint_config.type.lower() == "redis":
        tracker_store = RedisTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "mongod":
        tracker_store = MongoTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "sql":
        tracker_store = SQLTrackerStore(
            domain=domain,
            host=endpoint_config.url,
            event_broker=event_broker,
            **endpoint_config.kwargs,
        )
    elif endpoint_config.type.lower() == "dynamo":
        tracker_store = DynamoTrackerStore(
            domain=domain, event_broker=event_broker, **endpoint_config.kwargs
        )
    else:
        tracker_store = _load_from_module_name_in_endpoint_config(
            domain, endpoint_config, event_broker
        )

    structlogger.debug(
        "tracker_store.create_tracker_store_from_endpoint_config",
        event_info=f"Connected to {tracker_store.__class__.__name__}.",
    )

    return tracker_store


def _load_from_module_name_in_endpoint_config(
    domain: Domain, store: EndpointConfig, event_broker: Optional[EventBroker] = None
) -> TrackerStore:
    """Initializes a custom tracker.

    Defaults to the InMemoryTrackerStore if the module path can not be found.

    Args:
        domain: defines the universe in which the assistant operates
        store: the specific tracker store
        event_broker: an event broker to publish events

    Returns:
        a tracker store from a specified type in a stores endpoint configuration
    """
    try:
        tracker_store_class = rasa.shared.utils.common.class_from_module_path(
            store.type
        )

        return tracker_store_class(
            host=store.url, domain=domain, event_broker=event_broker, **store.kwargs
        )
    except (AttributeError, ImportError):
        rasa.shared.utils.io.raise_warning(
            f"Tracker store with type '{store.type}' not found. "
            f"Using `InMemoryTrackerStore` instead."
        )
        return InMemoryTrackerStore(domain)


def create_tracker_store(
    endpoint_config: Optional[EndpointConfig],
    domain: Optional[Domain] = None,
    event_broker: Optional[EventBroker] = None,
) -> TrackerStore:
    """Creates a tracker store based on the current configuration."""
    tracker_store = _create_from_endpoint_config(endpoint_config, domain, event_broker)

    if not check_if_tracker_store_async(tracker_store):
        rasa.shared.utils.io.raise_deprecation_warning(
            f"Tracker store implementation "
            f"{tracker_store.__class__.__name__} "
            f"is not asynchronous. Non-asynchronous tracker stores "
            f"are currently deprecated and will be removed in 4.0. "
            f"Please make the following methods async: "
            f"{_get_async_tracker_store_methods()}"
        )
        tracker_store = AwaitableTrackerStore(tracker_store)

    return tracker_store


class AwaitableTrackerStore(TrackerStore):
    """Wraps a tracker store so it can be implemented with async overrides."""

    def __init__(
        self,
        tracker_store: TrackerStore,
    ) -> None:
        """Create a `AwaitableTrackerStore`.

        Args:
            tracker_store: the wrapped tracker store.
        """
        self._tracker_store = tracker_store

        super().__init__(tracker_store.domain, tracker_store.event_broker)

    @property
    def domain(self) -> Domain:
        """Returns the domain of the primary tracker store."""
        return self._tracker_store.domain

    @domain.setter
    def domain(self, domain: Optional[Domain]) -> None:
        """Setter method to modify the wrapped tracker store's domain field."""
        self._tracker_store.domain = domain or Domain.empty()

    @staticmethod
    def create(
        obj: Union[TrackerStore, EndpointConfig, None],
        domain: Optional[Domain] = None,
        event_broker: Optional[EventBroker] = None,
    ) -> TrackerStore:
        """Wrapper to call `create` method of primary tracker store."""
        if isinstance(obj, TrackerStore):
            return AwaitableTrackerStore(obj)
        elif isinstance(obj, EndpointConfig):
            return AwaitableTrackerStore(_create_from_endpoint_config(obj))
        else:
            raise ValueError(
                f"{type(obj).__name__} supplied "
                f"but expected object of type {TrackerStore.__name__} or "
                f"of type {EndpointConfig.__name__}."
            )

    async def retrieve(self, sender_id: Text) -> Optional[DialogueStateTracker]:
        """Wrapper to call `retrieve` method of primary tracker store."""
        result = self._tracker_store.retrieve(sender_id)
        return await result if isawaitable(result) else result

    async def keys(self) -> Iterable[Text]:
        """Wrapper to call `keys` method of primary tracker store."""
        result = self._tracker_store.keys()
        return await result if isawaitable(result) else result

    async def save(self, tracker: DialogueStateTracker) -> None:
        """Wrapper to call `save` method of primary tracker store."""
        result = self._tracker_store.save(tracker)
        return await result if isawaitable(result) else result

    async def delete(self, sender_id: Text) -> None:
        """Delete tracker for the given sender_id."""
        result = self._tracker_store.delete(sender_id)
        return await result if isawaitable(result) else result

    async def update(self, tracker: DialogueStateTracker) -> None:
        """Replace an existing tracker with a new one."""
        result = self._tracker_store.update(tracker)
        return await result if isawaitable(result) else result

    async def retrieve_full_tracker(
        self, conversation_id: Text
    ) -> Optional[DialogueStateTracker]:
        """Wrapper to call `retrieve_full_tracker` method of primary tracker store."""
        result = self._tracker_store.retrieve_full_tracker(conversation_id)
        return await result if isawaitable(result) else result
