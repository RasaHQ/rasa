import itertools
import logging
import uuid
from typing import Text, Optional, List, Set, Dict, Any

from tqdm import tqdm

import rasa.cli.utils as cli_utils
import rasa.shared.utils.cli
from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.pika import PikaEventBroker
from rasa.core.constants import RASA_EXPORT_PROCESS_ID_HEADER_NAME
from rasa.core.tracker_store import TrackerStore
from rasa.shared.core.trackers import EventVerbosity
from rasa.exceptions import (
    NoEventsToMigrateError,
    NoConversationsInTrackerStoreError,
    NoEventsInTimeRangeError,
    PublishingError,
)

logger = logging.getLogger(__name__)


class Exporter:
    """Manages the publishing of events in a tracker store to an event broker.

    Attributes:
        endpoints_path: Path to the endpoints file used to configure the event
            broker and tracker store. If `None`, the default path ('endpoints.yml')
            is used.
        tracker_store: `TrackerStore` to export conversations from.
        event_broker: `EventBroker` to export conversations to.
        requested_conversation_ids: List of conversation IDs requested to be
            processed.
        minimum_timestamp: Minimum timestamp of events that are published.
            If `None`, apply no such constraint.
        maximum_timestamp: Maximum timestamp of events that are published.
            If `None`, apply no such constraint.
    """

    def __init__(
        self,
        tracker_store: TrackerStore,
        event_broker: EventBroker,
        endpoints_path: Text,
        requested_conversation_ids: Optional[Text] = None,
        minimum_timestamp: Optional[float] = None,
        maximum_timestamp: Optional[float] = None,
    ) -> None:
        self.endpoints_path = endpoints_path
        self.tracker_store = tracker_store
        # The `TrackerStore` should return all events on `retrieve` and not just the
        # ones from the last session.
        self.tracker_store.load_events_from_previous_conversation_sessions = True

        self.event_broker = event_broker
        self.requested_conversation_ids = requested_conversation_ids
        self.minimum_timestamp = minimum_timestamp
        self.maximum_timestamp = maximum_timestamp

    def publish_events(self) -> int:
        """Publish events in a tracker store using an event broker.

        Exits if the publishing of events is interrupted due to an error. In that case,
        the CLI command to continue the export where it was interrupted is printed.

        Returns:
            The number of successfully published events.

        """
        events = self._fetch_events_within_time_range()

        rasa.shared.utils.cli.print_info(
            f"Selected {len(events)} events for publishing. Ready to go 🚀"
        )

        published_events = 0
        current_timestamp = None

        headers = self._get_message_headers()

        for event in tqdm(events, "events"):
            # noinspection PyBroadException
            try:
                self._publish_with_message_headers(event, headers)
                published_events += 1
                current_timestamp = event["timestamp"]
            except Exception as e:
                logger.exception(e)
                raise PublishingError(current_timestamp)

        self.event_broker.close()

        return published_events

    def _get_message_headers(self) -> Optional[Dict[Text, Text]]:
        """Generate a message header for publishing events to a `PikaEventBroker`.

        Returns:
            Message headers with a randomly generated uuid under the
            `RASA_EXPORT_PROCESS_ID_HEADER_NAME` key if `self.event_broker` is a
            `PikaEventBroker`, else `None`.

        """
        if isinstance(self.event_broker, PikaEventBroker):
            return {RASA_EXPORT_PROCESS_ID_HEADER_NAME: uuid.uuid4().hex}

        return None

    def _publish_with_message_headers(
        self, event: Dict[Text, Any], headers: Optional[Dict[Text, Text]]
    ) -> None:
        """Publish `event` to a message broker with `headers`.

        Args:
            event: Serialized event to be published.
            headers: Message headers to be published if `self.event_broker` is a
                `PikaEventBroker`.

        """
        if isinstance(self.event_broker, PikaEventBroker):
            self.event_broker.publish(event=event, headers=headers)
        else:
            self.event_broker.publish(event)

    def _get_conversation_ids_in_tracker(self) -> Set[Text]:
        """Fetch conversation IDs in `self.tracker_store`.

        Returns:
            A set of conversation IDs in `self.tracker_store`.

        Raises:
            `NoConversationsInTrackerStoreError` if
            `conversation_ids_in_tracker_store` is empty.

        """
        conversation_ids_in_tracker_store = set(self.tracker_store.keys())

        if conversation_ids_in_tracker_store:
            return conversation_ids_in_tracker_store

        raise NoConversationsInTrackerStoreError(
            "Could not find any conversations in connected tracker store. "
            "Please validate your `endpoints.yml` and make sure the defined "
            "tracker store exists. Exiting."
        )

    def _validate_all_requested_ids_exist(
        self, conversation_ids_in_tracker_store: Set[Text]
    ) -> None:
        """Warn user if `self.requested_conversation_ids` contains IDs not found in
        `conversation_ids_in_tracker_store`

        Args:
            conversation_ids_in_tracker_store: Set of conversation IDs contained in
            the tracker store.

        """
        missing_ids_in_tracker_store = (
            set(self.requested_conversation_ids) - conversation_ids_in_tracker_store
        )
        if missing_ids_in_tracker_store:
            rasa.shared.utils.cli.print_warning(
                f"Could not find the following requested "
                f"conversation IDs in connected tracker store: "
                f"{', '.join(sorted(missing_ids_in_tracker_store))}"
            )

    def _get_conversation_ids_to_process(self) -> Set[Text]:
        """Get conversation IDs that are good for processing.

        Finds the intersection of events that are contained in the tracker store with
        those events requested as a command-line argument.

        Returns:
            Conversation IDs that are both requested and contained in the tracker
            store. If no conversation IDs are requested, all conversation IDs in the
            tracker store are returned.

        """
        conversation_ids_in_tracker_store = self._get_conversation_ids_in_tracker()

        if not self.requested_conversation_ids:
            return conversation_ids_in_tracker_store

        self._validate_all_requested_ids_exist(conversation_ids_in_tracker_store)

        conversation_ids_to_process = conversation_ids_in_tracker_store & set(
            self.requested_conversation_ids
        )

        if not conversation_ids_to_process:
            raise NoEventsToMigrateError(
                "Could not find an overlap between the requested "
                "conversation IDs and those found in the tracker store. Exiting."
            )

        return conversation_ids_to_process

    def _fetch_events_within_time_range(self) -> List[Dict[Text, Any]]:
        """Fetch all events for `conversation_ids` within the supplied time range.

        Returns:
            Serialized events with added `sender_id` field.

        """
        conversation_ids_to_process = self._get_conversation_ids_to_process()

        rasa.shared.utils.cli.print_info(
            f"Fetching events for {len(conversation_ids_to_process)} "
            f"conversation IDs:"
        )

        events = []

        for conversation_id in tqdm(conversation_ids_to_process, "conversation IDs"):
            tracker = self.tracker_store.retrieve(conversation_id)
            if not tracker:
                logger.info(
                    f"Could not retrieve tracker for conversation ID "
                    f"'{conversation_id}'. Skipping."
                )
                continue

            _events = tracker.current_state(EventVerbosity.ALL)["events"]

            if not _events:
                logger.info(
                    f"No events to migrate for conversation ID '{conversation_id}'."
                )
                continue

            # the conversation IDs are needed in the event publishing
            events.extend(
                self._get_events_for_conversation_id(_events, conversation_id)
            )

        return self._sort_and_select_events_by_timestamp(events)

    @staticmethod
    def _get_events_for_conversation_id(
        events: List[Dict[Text, Any]], conversation_id: Text
    ) -> List[Dict[Text, Any]]:
        """Get serialised events with added `sender_id` key.

        Args:
            events: Events to modify.
            conversation_id: Conversation ID to add to events.

        Returns:
            Events with added `sender_id` key.

        """
        events_with_conversation_id = []

        for event in events:
            event["sender_id"] = conversation_id
            events_with_conversation_id.append(event)

        return events_with_conversation_id

    def _sort_and_select_events_by_timestamp(
        self, events: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """Sort list of events by ascending timestamp, and select events within time
        range.

        Args:
            events: List of serialized events to be sorted and selected from.

        Returns:
            List of serialized and sorted (by timestamp) events within the requested
            time range.

        Raises:
             `NoEventsInTimeRangeError` error if no events are found within the
             requested time range.

        """
        logger.debug(f"Sorting and selecting from {len(events)} total events found.")
        # sort the events by timestamp just in case they're not sorted already
        events = sorted(events, key=lambda x: x["timestamp"])

        # drop events failing minimum timestamp requirement
        if self.minimum_timestamp is not None:
            events = itertools.dropwhile(
                lambda x: x["timestamp"] < self.minimum_timestamp, events
            )

        # select events passing maximum timestamp requirement
        if self.maximum_timestamp is not None:
            events = itertools.takewhile(
                lambda x: x["timestamp"] < self.maximum_timestamp, events
            )

        events = list(events)
        if not events:
            raise NoEventsInTimeRangeError(
                "Could not find any events within requested time range. Exiting."
            )

        return events
