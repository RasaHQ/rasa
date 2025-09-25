from __future__ import annotations

import asyncio
import copy
import datetime
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import structlog
from apscheduler.schedulers.background import BackgroundScheduler

import rasa.shared.core.trackers
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.privacy.constants import (
    TEXT_KEY,
    USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME,
)
from rasa.privacy.event_broker_utils import create_event_brokers
from rasa.privacy.privacy_config import (
    PrivacyConfig,
    PrivacyPolicy,
    validate_sensitive_slots,
)
from rasa.privacy.privacy_filter import PrivacyFilter
from rasa.shared.core.events import Event, SlotSet, UserUttered, split_events
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

    from rasa.core.brokers.broker import EventBroker
    from rasa.core.config.available_endpoints import AvailableEndpoints
    from rasa.shared.core.domain import Domain


structlogger = structlog.get_logger(__name__)


def wrap_async(func: Callable) -> Callable:
    """Wraps a function to be used as an async job in the background scheduler."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class BackgroundPrivacyManager:
    """Manages privacy-related tasks in the background.

    This class handles the anonymization and deletion of sensitive information
    in dialogue state trackers, as well as the streaming of anonymized events
    to event brokers. It uses background schedulers to periodically run these
    tasks and processes trackers from a queue to ensure that sensitive information
    is handled in a timely manner.
    """

    TRACKER_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS = 2.0

    def __init__(
        self,
        endpoints: Optional["AvailableEndpoints"],
        event_loop: Optional["AbstractEventLoop"] = None,
        in_memory_tracker_store: Optional[TrackerStore] = None,
    ):
        self.config = (
            PrivacyConfig.from_dict(endpoints.privacy)
            if endpoints and endpoints.privacy
            else None
        )
        self.privacy_filter = (
            PrivacyFilter(self.config.anonymization_rules) if self.config else None
        )
        self.user_chat_inactivity_in_minutes = int(
            os.getenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, 30)
        )

        if in_memory_tracker_store is not None:
            # if an in-memory tracker store is provided,
            # we need to keep the reference to it
            # so that the background jobs can access it.
            # We also set the event broker to None
            # to prevent it from publishing events
            # during the tracker store background jobs
            in_memory_tracker_store.event_broker = None
            tracker_store = in_memory_tracker_store
        else:
            # we recreate the tracker store here to ensure
            # that this instance has no event brokers
            # that could publish events during the tracker store
            # background jobs
            tracker_store = (
                TrackerStore.create(endpoints.tracker_store)
                if endpoints
                else TrackerStore.create(None)
            )

        self.tracker_store = tracker_store

        self.event_brokers: List["EventBroker"] = []
        self.event_loop = event_loop

        # Order of the initialisation is important
        # The tracker queue must be created before the scheduler
        # The can_consume_tracker_queue must be set to True before the scheduler starts
        self.tracker_queue: queue.Queue = queue.Queue()

        # This flag is used to stop the scheduler
        self.can_consume_from_tracker_queue = True
        self.background_scheduler = BackgroundScheduler()
        self.background_scheduler.add_job(
            self._consumer_queue, max_instances=1, id="event_broker_job"
        )

        self.previous_fire_time_deletion = datetime.datetime.now(
            tz=datetime.timezone.utc
        )
        self._configure_background_scheduler()
        self.background_scheduler.start()

    async def initialize(
        self, endpoints: Optional["AvailableEndpoints"]
    ) -> BackgroundPrivacyManager:
        """Initialize async attributes of the BackgroundPrivacyManager."""
        event_broker_endpoints = endpoints.event_broker if endpoints else None
        self.event_brokers = (
            await create_event_brokers(event_broker_endpoints, self.event_loop)
            if event_broker_endpoints
            else []
        )

        return self

    @classmethod
    async def create_instance(
        cls,
        endpoints: Optional["AvailableEndpoints"],
        event_loop: Optional["AbstractEventLoop"] = None,
        in_memory_tracker_store: Optional[TrackerStore] = None,
    ) -> BackgroundPrivacyManager:
        """Create an instance of BackgroundPrivacyManager."""
        instance = cls(endpoints, event_loop, in_memory_tracker_store)
        return await instance.initialize(endpoints)

    def stop(self) -> None:
        structlogger.debug("rasa.privacy_manager.stop_schedulers")
        self.can_consume_from_tracker_queue = False
        self.background_scheduler.shutdown(wait=False)

    def run(self, tracker: "DialogueStateTracker") -> None:
        self.tracker_queue.put(tracker)

    def process(
        self, tracker: "DialogueStateTracker", process_all: bool = False
    ) -> None:
        """Process the tracker to identify and anonymize sensitive information.

        Args:
            tracker: The tracker to process.
            process_all: If True, process all events in the tracker.
        """
        events = self.process_events(tracker, process_all=process_all)
        events_to_stream = events if events else tracker.events
        self.stream_events(events_to_stream, tracker.sender_id)

    def process_events(
        self, tracker: DialogueStateTracker, process_all: bool = False
    ) -> List[Event]:
        """Anonymize tracker events."""
        if (latest_message := self._get_latest_user_message(tracker)) is None:
            return []

        processed_events = list(tracker.events)
        prior_sensitive_slot_events: List[Event] = []

        if not process_all:
            additional_splitting_conditions = {
                TEXT_KEY: latest_message.text,
                "timestamp": latest_message.timestamp,
            }

            resulting_events = split_events(
                processed_events,
                UserUttered,
                additional_splitting_conditions=additional_splitting_conditions,
                include_splitting_event=True,
            )

            processed_events = resulting_events[1]
            prior_events = resulting_events[0]
            prior_tracker = DialogueStateTracker.from_events(
                sender_id=tracker.sender_id, evts=prior_events
            )
            prior_sensitive_slot_events = [
                event
                for event in prior_tracker.applied_events()
                if isinstance(event, SlotSet)
                and event.key in self.config.anonymization_rules  # type: ignore[union-attr]
            ]

        return self.privacy_filter.anonymize(  # type: ignore[union-attr]
            processed_events, prior_sensitive_slot_events
        )

    def stream_events(
        self,
        anonymized_events: List[Event],
        sender_id: str,
    ) -> None:
        """Stream anonymized events to the event broker."""
        if not self.event_brokers:
            structlogger.debug(
                "rasa.privacy_manager.no_event_broker_configured",
            )
            return None

        for event in anonymized_events:
            body = {"sender_id": sender_id}
            body.update(event.as_dict())
            for broker in self.event_brokers:
                broker.publish(body)

        return None

    def validate_sensitive_slots_in_domain(self, domain: "Domain") -> None:
        """Validate the sensitive slots defined in the privacy config against the domain."""  # noqa: E501
        if not self.config:
            structlogger.debug(
                "rasa.privacy_manager.no_sensitive_slots_configured",
            )
            return None

        # we need to set the domain in the tracker store
        # to prevent errors being raised about slots not found in the domain
        # during the background jobs
        self.tracker_store.domain = domain
        sensitive_slots = list(self.config.anonymization_rules.keys())
        return validate_sensitive_slots(sensitive_slots, domain)

    def _consumer_queue(self) -> None:
        while self.can_consume_from_tracker_queue:
            try:
                # Wait for 2 seconds for an event to be added to the queue
                # If no event is added to the queue, continue
                # This is done to avoid the scheduler to be stuck in the while loop
                # when we want to stop the scheduler
                tracker = self.tracker_queue.get(
                    timeout=self.TRACKER_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS
                )
                self.process(tracker)
                self.tracker_queue.task_done()
            except queue.Empty:
                continue

    def _get_latest_user_message(
        self, tracker: DialogueStateTracker
    ) -> Optional[UserUttered]:
        """Check if a tracker should be processed."""
        if self.privacy_filter is None:
            structlogger.debug(
                "rasa.privacy_manager.no_privacy_rules_configured",
            )
            return None

        latest_user_message_event = tracker.get_last_event_for(
            UserUttered, event_verbosity=EventVerbosity.ALL
        )
        if latest_user_message_event is None or not isinstance(
            latest_user_message_event, UserUttered
        ):
            structlogger.debug(
                "rasa.privacy_manager.no_user_message.skipping_processing",
            )
            return None

        latest_user_message: UserUttered = latest_user_message_event
        if not latest_user_message.text:
            structlogger.debug(
                "rasa.privacy_manager.no_user_message.skipping_processing",
            )
            return None

        return latest_user_message

    @staticmethod
    def _has_session_been_anonymized(events: List[Event]) -> bool:
        """Check if the session has already been anonymized."""
        if not events:
            return False
        for event in reversed(events):
            if (
                hasattr(event, "anonymized_at")
                and getattr(event, "anonymized_at") is not None
            ):
                return True

        return False

    async def _run_tracker_store_anonymization(self) -> None:
        """Anonymize eligible tracker sessions in the tracker store."""
        structlogger.info(
            "rasa.privacy_manager.starting_tracker_store_anonymization",
            triggered_by="anonymization_cron_job",
        )

        keys = await self.tracker_store.keys()
        keys_copy = copy.deepcopy(list(keys))

        for key in keys_copy:
            full_tracker = await self.tracker_store.retrieve_full_tracker(key)

            if not full_tracker:
                structlogger.debug(
                    "rasa.privacy_manager.no_tracker_found_for_sender_id",
                    sender_id=key,
                )
                continue

            processed_events, already_anonymized_events, uneligible_events = (
                self._get_processed_events_after_anonymization(full_tracker)
            )

            if not processed_events:
                structlogger.debug(
                    "rasa.privacy_manager.no_events_to_anonymize_for_tracker",
                    sender_id=key,
                )
                continue

            all_events = (
                already_anonymized_events + processed_events + uneligible_events
            )
            updated_tracker = DialogueStateTracker.from_events(
                sender_id=key,
                evts=all_events,
                slots=full_tracker.slots.values(),
            )
            await self.tracker_store.delete(sender_id=key)
            await self.tracker_store.save(updated_tracker)

            structlogger.info(
                "rasa.privacy_manager.saved_tracker_after_anonymization",
                sender_id=key,
            )

    async def _run_tracker_store_deletion(self) -> None:
        """Delete eligible tracker sessions from the tracker store."""
        structlogger.info(
            "rasa.privacy_manager.starting_tracker_store_deletion",
            triggered_by="deletion_cron_job",
        )
        keys = await self.tracker_store.keys()

        # Make a copy of the keys to avoid modifying the list while iterating
        keys_copy = copy.deepcopy(list(keys))

        for key in keys_copy:
            full_tracker = await self.tracker_store.retrieve_full_tracker(key)

            if not full_tracker:
                structlogger.debug(
                    "rasa.privacy_manager.no_tracker_found_for_sender_id",
                    sender_id=key,
                )
                continue

            events_to_be_retained = self._get_events_to_be_retained_after_deletion(
                full_tracker
            )

            if not events_to_be_retained:
                await self.tracker_store.delete(sender_id=key)
                structlogger.info(
                    "rasa.privacy_manager.tracker_session_deleted",
                    sender_id=full_tracker.sender_id,
                    triggered_by="deletion_cron_job",
                )
                continue

            tracker = DialogueStateTracker.from_events(
                sender_id=key,
                evts=events_to_be_retained,
                slots=full_tracker.slots.values(),
            )
            await self.tracker_store.update(tracker)

            structlogger.info(
                "rasa.privacy_manager.overwritten_tracker",
                sender_id=key,
                event_info="Deleted eligible events and saved "
                "tracker with events not scheduled "
                "for deletion yet.",
            )

    async def _run_tracker_store_background_jobs_sequentially(self) -> None:
        """Run the tracker store background jobs.

        If both anonymization and deletion policies are configured,
        we need to ensure that the background job timings do not
        overlap to prevent race conditions when accessing the
        tracker store.

        The scheduler will run the anonymization job first,
        and then the deletion job if the current time is past
        the next scheduled time for deletion.
        """
        await self._run_tracker_store_anonymization()

        now = datetime.datetime.now(tz=datetime.timezone.utc)
        next_fire_time = (
            self.config.tracker_store_settings.deletion_policy.cron.get_next_fire_time(  # type: ignore[union-attr]
                self.previous_fire_time_deletion,
                now=now,
            )
        )

        if next_fire_time and now >= next_fire_time:
            await self._run_tracker_store_deletion()
            self.previous_fire_time_deletion = next_fire_time

        return None

    def _add_anonymization_job(self) -> None:
        wrapped_anonymization = wrap_async(self._run_tracker_store_anonymization)
        self.background_scheduler.add_job(
            wrapped_anonymization,
            trigger=self.config.tracker_store_settings.anonymization_policy.cron,  # type: ignore[union-attr]
            max_instances=1,
            id="anonymization_cron_job",
        )

    def _add_deletion_job(self) -> None:
        wrapped_deletion = wrap_async(self._run_tracker_store_deletion)
        self.background_scheduler.add_job(
            wrapped_deletion,
            trigger=self.config.tracker_store_settings.deletion_policy.cron,  # type: ignore[union-attr]
            max_instances=1,
            id="deletion_cron_job",
        )

    def _add_sequential_job(self) -> None:
        sequential_dispatcher = wrap_async(
            self._run_tracker_store_background_jobs_sequentially
        )
        self.background_scheduler.add_job(
            sequential_dispatcher,
            trigger=self.config.tracker_store_settings.anonymization_policy.cron,  # type: ignore[union-attr]
            max_instances=1,
            id="anonymization_and_deletion_cron_job",
        )

    def _configure_background_scheduler(self) -> None:
        """Configure the background scheduler."""
        tracker_store_settings_configured = (
            self.config is not None and self.config.tracker_store_settings is not None
        )
        anonymization_policy = (
            self.config.tracker_store_settings.anonymization_policy  # type: ignore[union-attr]
            if tracker_store_settings_configured
            else None
        )
        deletion_policy = (
            self.config.tracker_store_settings.deletion_policy  # type: ignore[union-attr]
            if tracker_store_settings_configured
            else None
        )

        if (
            tracker_store_settings_configured
            and anonymization_policy is not None
            and deletion_policy is not None
        ):
            next_fire_time_anonymization = get_next_fire_time(anonymization_policy)
            next_fire_time_deletion = get_next_fire_time(deletion_policy)

            # If both anonymization and deletion policies are configured
            # to start on the same date,
            # we need to run them sequentially to avoid race conditions
            if (
                next_fire_time_anonymization is not None
                and next_fire_time_deletion is not None
                and next_fire_time_anonymization.date()
                == next_fire_time_deletion.date()
            ):
                self._add_sequential_job()
            else:
                self._add_anonymization_job()
                self._add_deletion_job()

        elif tracker_store_settings_configured and anonymization_policy is not None:
            self._add_anonymization_job()

        elif tracker_store_settings_configured and deletion_policy is not None:
            self._add_deletion_job()

    def _get_processed_events_after_anonymization(
        self,
        full_tracker: DialogueStateTracker,
    ) -> Tuple[List[Event], List[Event], List[Event]]:
        """Get processed events after anonymization job."""
        multiple_tracker_sessions = (
            rasa.shared.core.trackers.get_trackers_for_conversation_sessions(
                full_tracker
            )
        )

        processed_events = []
        already_anonymized_events = []
        uneligible_events = []

        for session in multiple_tracker_sessions:
            has_session_been_anonymized = self._has_session_been_anonymized(
                list(session.events)
            )

            if has_session_been_anonymized:
                structlogger.debug(
                    "rasa.privacy_manager.session_already_anonymized",
                    session_id=session.sender_id,
                )
                already_anonymized_events.extend(list(session.events))
                continue

            current_time = time.time()

            last_event_timestamp = (
                str(datetime.datetime.fromtimestamp(session.events[-1].timestamp))
                if session.events
                else "N/A"
            )

            if session.events and current_time - session.events[-1].timestamp > (
                self.user_chat_inactivity_in_minutes * 60
                + self.config.tracker_store_settings.anonymization_policy.min_after_session_end  # type: ignore[union-attr] # noqa: E501
                * 60
            ):
                structlogger.info(
                    "rasa.privacy_manager.anonymizing_tracker_session",
                    sender_id=session.sender_id,
                    last_event_timestamp=last_event_timestamp,
                    triggered_by="anonymization_cron_job",
                )
                events = self.process_events(session, process_all=True)
                processed_events.extend(events)
            else:
                # If the session is not valid for anonymization,
                # we still want to write them back to the tracker store
                events = list(session.events)
                uneligible_events.extend(events)
                structlogger.debug(
                    "rasa.privacy_manager.session_not_valid_for_anonymization",
                    sender_id=session.sender_id,
                    session_id=session.sender_id,
                    last_event_timestamp=last_event_timestamp,
                )
        return processed_events, already_anonymized_events, uneligible_events

    def _get_events_to_be_retained_after_deletion(
        self, full_tracker: DialogueStateTracker
    ) -> List[Event]:
        """Get the events to be retained after deletion."""
        multiple_tracker_sessions = (
            rasa.shared.core.trackers.get_trackers_for_conversation_sessions(
                full_tracker
            )
        )
        events_to_be_retained: List[Event] = []
        for session in multiple_tracker_sessions:
            current_time = time.time()
            if session.events and (
                current_time - session.events[-1].timestamp
                <= (
                    self.user_chat_inactivity_in_minutes * 60
                    + self.config.tracker_store_settings.deletion_policy.min_after_session_end  # type: ignore[union-attr] # noqa: E501
                    * 60
                )
            ):
                events_to_be_retained.extend(session.events)
            else:
                last_event_timestamp = (
                    str(datetime.datetime.fromtimestamp(session.events[-1].timestamp))
                    if session.events
                    else "N/A"
                )

                structlogger.info(
                    "rasa.privacy_manager.tracker_session_scheduled_for_deletion",
                    sender_id=full_tracker.sender_id,
                    last_event_timestamp=last_event_timestamp,
                    triggered_by="deletion_cron_job",
                )

        return events_to_be_retained


def get_next_fire_time(
    privacy_policy: PrivacyPolicy,
) -> Optional[datetime.datetime]:
    """Get the next fire time for the privacy policy."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return privacy_policy.cron.get_next_fire_time(None, now=now)
