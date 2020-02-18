import itertools
import logging
from typing import Text, Optional, List, Set, Dict, Any

from tqdm import tqdm

import rasa.cli.utils as cli_utils
from rasa.core.brokers.broker import EventBroker
from rasa.core.tracker_store import TrackerStore
from rasa.core.trackers import EventVerbosity
from rasa.exceptions import (
    NoEventsToMigrate,
    NoConversationsInTrackerStore,
    NoEventsInTimeRange,
    PublishingError,
)

logger = logging.getLogger(__name__)


class Migrator:
    """Manages the publishing of events in a tracker store to an event broker."""

    def __init__(
        self,
        tracker_store: TrackerStore,
        event_broker: EventBroker,
        endpoints_path: Text,
        requested_conversation_ids: Optional[Text] = None,
        minimum_timestamp: Optional[float] = None,
        maximum_timestamp: Optional[float] = None,
    ):
        """
        Args:
            endpoints_path: Path to the endpoints file used to configure the event
                broker and tracker store. If `None`, the default path ('endpoints.yml')
                is used.
            requested_conversation_ids: List of conversation IDs requested to be
                processed.
            minimum_timestamp: Minimum timestamp of events that are published.
                If `None`, apply no such constraint.
            maximum_timestamp: Maximum timestamp of events that are published.
                If `None`, apply no such constraint.

        """
        self.endpoints_path = endpoints_path
        self.tracker_store = tracker_store
        self.event_broker = event_broker
        self.requested_conversation_ids = requested_conversation_ids
        self.minimum_timestamp = minimum_timestamp
        self.maximum_timestamp = maximum_timestamp

    def publish_events(self) -> int:
        """Publish events in a tracker store using an event broker.

        Exits if the publishing of events is interrupted due to an error. In that case,
        the CLI command to continue the export where it was interrupted is printed.

        """
        events = self._fetch_events_within_time_range()

        cli_utils.print_info(
            f"Selected {len(events)} events for publishing. Ready to go 🚀"
        )

        published_events = 0
        current_timestamp = None

        for event in tqdm(events, "events"):
            # noinspection PyBroadException
            try:
                body = {"sender_id": event["sender_id"]}
                body.update(event)
                self.event_broker.publish(body)
                published_events += 1
                current_timestamp = event["timestamp"]
            except Exception as e:
                logger.exception(e)
                raise PublishingError(current_timestamp)

        self.event_broker.close()

        return published_events

    def get_conversation_ids_to_process(self) -> Set[Text]:
        """Get conversation IDs that are good for processing.

        Finds the intersection of events that are contained in the tracker store with
        those events requested as a command-line argument.

        Prints an error and if no conversation IDs are found in the tracker, or if no
        overlap is found between those contained in the tracker and those requested
        by the user.

        Returns:
            Conversation IDs that are both requested and contained in the tracker
            store. If no conversation IDs are requested, all conversation IDs in the
            tracker store are returned.

        """
        conversation_ids_in_tracker_store = set(self.tracker_store.keys())

        if not conversation_ids_in_tracker_store:
            raise NoConversationsInTrackerStore(
                f"Could not find any conversations in connected tracker store. "
                f"Please validate your `endpoints.yml` and make sure the defined "
                f"tracker store exists. Exiting."
            )

        if not self.requested_conversation_ids:
            return conversation_ids_in_tracker_store

        missing_ids_in_tracker_store = (
            set(self.requested_conversation_ids) - conversation_ids_in_tracker_store
        )

        if missing_ids_in_tracker_store:
            cli_utils.print_warning(
                f"Could not find the following requested "
                f"conversation IDs in connected tracker store: "
                f"{', '.join(sorted(missing_ids_in_tracker_store))}"
            )

        conversation_ids_to_process = conversation_ids_in_tracker_store & set(
            self.requested_conversation_ids
        )

        if not conversation_ids_to_process:
            raise NoEventsToMigrate(
                "Could not find an overlap between the requested "
                "conversation IDs and those found in the tracker store. Exiting."
            )

        return conversation_ids_to_process

    def _fetch_events_within_time_range(self) -> List[Dict[Text, Any]]:
        """Fetch all events for `conversation_ids` within the supplied time range.

        Returns:
            List of serialized events with added `sender_id` field.

        """
        conversation_ids_to_process = self.get_conversation_ids_to_process()

        cli_utils.print_info(
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
            for event in _events:
                event["sender_id"] = conversation_id
                events.append(event)

        return self._sort_and_select_events_by_timestamp(events)

    def _sort_and_select_events_by_timestamp(
        self, events: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """Sort list of events by ascending timestamp, and select events within time range.

        Raises `NoEventsInTimeRange` error if no events are found within the requested
        time range.

        Args:
            events: List of serialized events to be sorted and selected from.

        Returns:
            List of serialized and sorted (by timestamp) events within the requested time
            range.

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
            raise NoEventsInTimeRange(
                "Could not find any events within requested time range. Exiting."
            )

        return events
