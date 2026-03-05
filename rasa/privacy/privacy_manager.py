from __future__ import annotations

import asyncio
import copy
import datetime
import os
import queue
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointerException

from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.privacy.constants import (
    ANONYMIZATION_LOG_KEY,
    DELETION_LOG_KEY,
    LEGACY_DEFAULT_INACTIVITY_MINUTES,
    NO_SESSION_ID_KEY,
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
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.events import (
    ActionExecuted,
    ConversationInactive,
    Event,
    SessionEnded,
    SlotSet,
    UserUttered,
    split_events,
)
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import METADATA_SESSION_ID
from rasa.shared.utils.io import raise_deprecation_warning

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

    from rasa.core.brokers.broker import EventBroker
    from rasa.core.config.available_endpoints import AvailableEndpoints
    from rasa.core.lock_store import LockStore
    from rasa.shared.core.domain import Domain


structlogger = structlog.get_logger(__name__)

# Session represented as (sender_id, events) when only event lists are needed.
SessionEvents = Tuple[str, List[Event]]


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

    Tracker variants and how they are handled:
    - Legacy trackers: No session_id in any event metadata. They can contain
      multiple ActionExecuted(action_session_start) events (e.g. after session
      expiry a new session starts). We split by action_session_start.
      Eligibility is time-based using
      LEGACY_DEFAULT_INACTIVITY_MINUTES + min_after_session_end.
    - New trackers: Events have session_id in metadata. They have
      ConversationInactive after session timeout; a subset also have
      SessionEnded (not all). Some have multiple action_session_start at the
      start of a session, some have only one. We group events by session_id
      (we do not split by action_session_start). A session is eligible for
      anonymization/deletion only if it contains ConversationInactive or
      SessionEnded; retention uses min_after_session_end after that event.
    - Hybrid (resumed): A tracker can have both legacy events (no session_id)
      and new events (with session_id). We assume legacy events form a single
      prefix: all events before a version upgrade have no session_id; after
      the upgrade all new events have session_id. So the order is [legacy
      prefix] then [session_id runs]. We take the event-only path (because at
      least one event has session_id). The legacy prefix is expanded via
      _get_legacy_session_events (split by action_session_start) and each sub-session
      is evaluated individually for anonymization; session_id runs are processed
      per run so reassembly preserves order.

    When USER_CHAT_INACTIVITY_IN_MINUTES is set, eligibility is time-based
    (env value + min_after_session_end). Session splitting still respects
    new/hybrid trackers: if any event has session_id we group by session_id,
    expand the NO_SESSION_ID_KEY segment via the legacy split, and apply the
    env-set threshold to each resulting session.
    """

    TRACKER_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS = 2.0

    def __init__(
        self,
        endpoints: Optional["AvailableEndpoints"],
        event_loop: Optional["AbstractEventLoop"] = None,
        in_memory_tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional["LockStore"] = None,
    ):
        if lock_store is None:
            raise RasaException(
                "LockStore is required for BackgroundPrivacyManager. "
                "Pass lock_store when creating the manager."
            )
        self.lock_store = lock_store
        self.config = (
            PrivacyConfig.from_dict(endpoints.privacy)
            if endpoints and endpoints.privacy
            else None
        )
        self.privacy_filter = (
            PrivacyFilter(self.config.anonymization_rules) if self.config else None
        )
        _inactivity_env = os.getenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME)
        if _inactivity_env is not None:
            try:
                self.user_chat_inactivity_in_minutes: Optional[int] = int(
                    _inactivity_env
                )
                raise_deprecation_warning(
                    f"Environment variable "
                    f"'{USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME}' "
                    "is deprecated for detecting inactive conversations "
                    "in privacy jobs. Use ConversationInactive or "
                    "SessionEnded events instead."
                )
            except (ValueError, TypeError):
                structlogger.warning(
                    "rasa.privacy_manager.invalid_env_value",
                    env_var=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME,
                    value=_inactivity_env,
                    event_info="USER_CHAT_INACTIVITY_IN_MINUTES must be an "
                    "integer representing minutes. Falling back "
                    "to event-based detection of inactive conversations.",
                )
                self.user_chat_inactivity_in_minutes = None
        else:
            self.user_chat_inactivity_in_minutes = None

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
        lock_store: Optional["LockStore"] = None,
    ) -> BackgroundPrivacyManager:
        """Create an instance of BackgroundPrivacyManager."""
        instance = cls(endpoints, event_loop, in_memory_tracker_store, lock_store)
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

    def process_events_from_segment(
        self,
        events: List[Event],
        prior_sensitive_slot_events: Optional[List[Event]] = None,
    ) -> List[Event]:
        """Anonymize a segment of events without replaying into a tracker.

        Use this when the segment may contain dialogue stack updates that
        assume a different stack state (e.g. segments from the middle of a
        run). Replaying such segments into a tracker can raise when applying
        stack patches. Passing the events directly avoids that.
        """
        if not events:
            return []
        return self.privacy_filter.anonymize(  # type: ignore[union-attr]
            events, prior_sensitive_slot_events or []
        )

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
            # Derive prior_sensitive_slot_events from raw prior_events instead of
            # replaying into a tracker (DialogueStateTracker.from_events). Replaying
            # can raise JsonPointerException when prior_events contain dialogue stack
            # patches that assume a different stack state (e.g. segments spanning
            # multiple sessions). Scanning events directly avoids that.
            prior_sensitive_slot_events = [
                event
                for event in prior_events
                if isinstance(event, SlotSet)
                and event.key in self.config.anonymization_rules  # type: ignore[union-attr]
            ]

        return self.process_events_from_segment(
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

    @staticmethod
    def _tracker_has_any_session_id(tracker: DialogueStateTracker) -> bool:
        """Return True if any event in the tracker has session_id in metadata."""
        for event in tracker.events:
            if event.metadata.get(METADATA_SESSION_ID):
                return True
        return False

    @staticmethod
    def _group_events_by_session_id(events: List[Event]) -> Dict[str, List[Event]]:
        """Group events by session_id from event metadata.

        Events with the same non-empty session_id are grouped together.
        Events without session_id in metadata are grouped under NO_SESSION_ID_KEY.
        Processor-emitted events (e.g. ConversationInactive) get session_id injected
        when the tracker is updated (DialogueStateTracker._prepare_event_metadata),
        so they do not create a separate NO_SESSION_ID_KEY group for live trackers.
        """
        grouped: Dict[str, List[Event]] = {}
        for event in events:
            session_id = event.metadata.get(METADATA_SESSION_ID)
            if session_id and isinstance(session_id, str):
                key = session_id.strip()
            else:
                key = NO_SESSION_ID_KEY
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(event)
        return grouped

    @staticmethod
    def _event_session_key(event: Event) -> str:
        """Return session key for event (NO_SESSION_ID_KEY or session_id string)."""
        session_id = event.metadata.get(METADATA_SESSION_ID)
        if session_id and isinstance(session_id, str):
            return session_id.strip()
        return NO_SESSION_ID_KEY

    @staticmethod
    def _group_events_into_runs(
        events: List[Event],
    ) -> List[Tuple[str, List[Event]]]:
        """Group events into runs for hybrid trackers, preserving order.

        Assumption: legacy events (no session_id in metadata) appear only as a
        single prefix, from trackers created before a version upgrade. After the
        upgrade, all new events have session_id. So the shape is [legacy prefix]
        then [one or more session_id runs] (e.g. s1, then s2 after /restart).

        We first take the legacy prefix (events with no session_id from the
        start). The remainder is split into consecutive runs by session_id so
        that reassembly preserves order and DialogueStateTracker.from_events
        replays correctly.
        """
        if not events:
            return []
        # Legacy prefix: events with no session_id from the start
        legacy_prefix: List[Event] = []
        i = 0
        while i < len(events) and (
            BackgroundPrivacyManager._event_session_key(events[i]) == NO_SESSION_ID_KEY
        ):
            legacy_prefix.append(events[i])
            i += 1
        runs: List[Tuple[str, List[Event]]] = []
        if legacy_prefix:
            runs.append((NO_SESSION_ID_KEY, legacy_prefix))
        if i >= len(events):
            return runs
        # Remainder: split into consecutive runs by session_id. Keep
        # ConversationInactive/SessionEnded in the same run as the previous
        # event when they have no session_id (processor emits them without
        # metadata), so we can anonymize the run when that marker is past grace.
        remainder = events[i:]
        current_key = BackgroundPrivacyManager._event_session_key(remainder[0])
        current_run: List[Event] = [remainder[0]]
        for event in remainder[1:]:
            key = BackgroundPrivacyManager._event_session_key(event)
            if key == NO_SESSION_ID_KEY and isinstance(
                event, (ConversationInactive, SessionEnded)
            ):
                key = current_key
            if key != current_key:
                runs.append((current_key, current_run))
                current_key = key
                current_run = [event]
            else:
                current_run.append(event)
        runs.append((current_key, current_run))
        return runs

    @staticmethod
    def _session_events_contain_inactive_or_ended(events: List[Event]) -> bool:
        """Return True if the event list contains ConversationInactive or SessionEnded."""  # noqa: E501
        return any(isinstance(e, (ConversationInactive, SessionEnded)) for e in events)

    @staticmethod
    def _split_events_by_inactive_or_ended(
        events: List[Event],
    ) -> List[List[Event]]:
        """Split events into segments ending at ConversationInactive or SessionEnded.

        Each segment is events from the previous split (or start) up to and
        including the next ConversationInactive or SessionEnded. The last
        segment may have no inactive/ended event at the end.
        """
        if not events:
            return []
        segments: List[List[Event]] = []
        current: List[Event] = []
        for event in events:
            current.append(event)
            if isinstance(event, (ConversationInactive, SessionEnded)):
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return segments

    @staticmethod
    def _iter_segments_with_grace_status(
        run_events: List[Event],
        min_after_seconds: float,
        current_time: float,
    ) -> List[Tuple[List[Event], bool]]:
        """Split run by inactive/ended; return (segment, past_grace) per segment.

        past_grace is True when the segment ends with ConversationInactive or
        SessionEnded and (current_time - segment_last_ts) > min_after_seconds.
        Used by event-only anonymization (process past_grace segments) and
        deletion (drop past_grace segments, retain the rest).
        """
        segments = BackgroundPrivacyManager._split_events_by_inactive_or_ended(
            run_events
        )
        result: List[Tuple[List[Event], bool]] = []
        for segment in segments:
            if not segment:
                continue
            seg_last_ts = segment[-1].timestamp
            seg_ends_with_inactive_or_ended = isinstance(
                segment[-1], (ConversationInactive, SessionEnded)
            )
            past_grace = (
                seg_ends_with_inactive_or_ended
                and (current_time - seg_last_ts) > min_after_seconds
            )
            result.append((segment, past_grace))
        return result

    def _get_legacy_session_events(
        self, sender_id: str, events: List[Event]
    ) -> List[SessionEvents]:
        """Return session (sender_id, events) lists without building trackers.

        Splits by action_session_start; we don't split by
        ConversationInactive/SessionEnded here because
        these events are not present in a legacy tracker.
        """
        split_conversations = split_events(
            events,
            ActionExecuted,
            {"action_name": ACTION_SESSION_START_NAME},
            include_splitting_event=True,
        )
        return [(sender_id, evts) for evts in split_conversations]

    def _get_legacy_threshold_seconds(
        self, job_type: Literal["anonymization", "deletion"]
    ) -> float:
        """Return time threshold in seconds for legacy path.

        LEGACY_DEFAULT_INACTIVITY_MINUTES + min_after_session_end.
        """
        if self.config is None or self.config.tracker_store_settings is None:
            raise RasaException(
                "Privacy config and tracker store settings are required "
                "for legacy threshold calculation."
            )
        policy = (
            self.config.tracker_store_settings.anonymization_policy
            if job_type == "anonymization"
            else self.config.tracker_store_settings.deletion_policy
        )
        return (
            LEGACY_DEFAULT_INACTIVITY_MINUTES * 60 + policy.min_after_session_end * 60  # type: ignore[union-attr]
        )

    def _get_env_set_threshold_seconds(
        self, job_type: Literal["anonymization", "deletion"]
    ) -> float:
        """Return time threshold when USER_CHAT_INACTIVITY_IN_MINUTES is set.

        env * 60 + min_after_session_end (seconds).
        """
        if self.config is None or self.config.tracker_store_settings is None:
            raise RasaException(
                "Privacy config and tracker store settings are required "
                "for env-set threshold calculation."
            )
        if self.user_chat_inactivity_in_minutes is None:
            raise RasaException(
                "USER_CHAT_INACTIVITY_IN_MINUTES must be set for env-set threshold."
            )
        policy = (
            self.config.tracker_store_settings.anonymization_policy
            if job_type == "anonymization"
            else self.config.tracker_store_settings.deletion_policy
        )
        inactivity = self.user_chat_inactivity_in_minutes * 60
        return inactivity + policy.min_after_session_end * 60  # type: ignore[union-attr]

    def _get_env_set_session_trackers(
        self, full_tracker: DialogueStateTracker
    ) -> List[SessionEvents]:
        """Return session (sender_id, events) when env is set; no trackers built.

        New/hybrid: use runs (consecutive same session_id) to preserve event order;
        NO_SESSION_ID_KEY runs split via _get_legacy_session_events.
        """
        if not self._tracker_has_any_session_id(full_tracker):
            return self._get_legacy_session_events(
                full_tracker.sender_id, list(full_tracker.events)
            )
        runs = self._group_events_into_runs(list(full_tracker.events))
        sessions: List[SessionEvents] = []
        for session_key, run_events in runs:
            if not run_events:
                continue
            if session_key == NO_SESSION_ID_KEY:
                sessions.extend(
                    self._get_legacy_session_events(full_tracker.sender_id, run_events)
                )
            else:
                sessions.append((full_tracker.sender_id, run_events))
        return sessions

    def _is_tracker_eligible_for_privacy_job(
        self,
        tracker: DialogueStateTracker,
        job_type: Literal["anonymization", "deletion"],
    ) -> bool:
        """Return True if the tracker is eligible for the given privacy cron job.

        Eligible when:
        - Event-based: tracker.inactive or tracker.terminated, or
        - Time-based (env set): last event older than
            user_chat_inactivity + min_after_session_end, or
        - Legacy (env unset, no session_id): last event older than
            LEGACY_DEFAULT_INACTIVITY + min_after_session_end.
        """
        if tracker.terminated:
            return True
        if job_type == "anonymization" and tracker.inactive:
            return True
        if not tracker.events:
            return False
        if self.config is None or self.config.tracker_store_settings is None:
            return False

        last_event_timestamp = tracker.events[-1].timestamp
        current_time = time.time()
        threshold_seconds: Optional[float] = None

        if self.user_chat_inactivity_in_minutes is not None:
            threshold_seconds = self._get_env_set_threshold_seconds(job_type)
        elif not self._tracker_has_any_session_id(tracker):
            threshold_seconds = self._get_legacy_threshold_seconds(job_type)

        # For deletion without env, we only actually delete when tracker is
        # terminated; multi-session and time-threshold shortcuts would still
        # retain all events (no-op) and block the lock every cron run.
        deletion_no_env = (
            job_type == "deletion" and self.user_chat_inactivity_in_minutes is None
        )

        if threshold_seconds is not None and self._tracker_has_multiple_sessions(
            tracker
        ):
            return not deletion_no_env

        if (
            threshold_seconds is not None
            and (current_time - last_event_timestamp) > threshold_seconds
        ):
            return not deletion_no_env

        return False

    def _tracker_has_multiple_sessions(self, tracker: DialogueStateTracker) -> bool:
        """Return True if the tracker would be split into more than one session.

        When True, per-session processing may still apply to some sessions even
        if the full tracker fails _is_tracker_eligible_for_privacy_job (e.g.
        latest session active but earlier sessions old or with inactive/ended).

        For trackers with session_id: grouping is by event metadata. Events
        without session_id in metadata are legacy (stored before injection);
        live-added events get session_id from the tracker on update, so they
        do not falsely create an extra group.
        """
        if not tracker.events:
            return False
        if self._tracker_has_any_session_id(tracker):
            grouped = self._group_events_by_session_id(list(tracker.events))
            return len(grouped) > 1
        sessions = self._get_legacy_session_events(
            tracker.sender_id, list(tracker.events)
        )
        return len(sessions) > 1

    async def _process_one_key_anonymization(self, key: str) -> None:
        """Process one tracker for anonymization (call under lock)."""
        full_tracker = await self.tracker_store.retrieve_full_tracker(key)
        if full_tracker is None:
            structlogger.debug(
                "rasa.privacy_manager.no_tracker_found_for_sender_id",
                sender_id=key,
            )
            return
        if not self._is_tracker_eligible_for_privacy_job(
            full_tracker, "anonymization"
        ) and not self._tracker_has_multiple_sessions(full_tracker):
            return
        all_events, num_processed = self._get_processed_events_after_anonymization(
            full_tracker
        )
        if num_processed == 0:
            structlogger.debug(
                "rasa.privacy_manager.no_events_to_anonymize_for_tracker",
                sender_id=key,
            )
            return
        updated_tracker = DialogueStateTracker.from_events(
            sender_id=key,
            evts=all_events,
            slots=full_tracker.slots.values(),
            user_id=full_tracker.user_id,
        )
        await self.tracker_store.update(updated_tracker, apply_deletion_only=False)
        structlogger.info(
            "rasa.privacy_manager.saved_tracker_after_anonymization",
            sender_id=key,
        )

    async def _run_tracker_store_anonymization(self) -> None:
        """Anonymize eligible tracker sessions in the tracker store."""
        structlogger.info(
            "rasa.privacy_manager.starting_tracker_store_anonymization",
            triggered_by="anonymization_cron_job",
        )

        keys = await self.tracker_store.keys()
        keys_copy = copy.deepcopy(list(keys))

        for key in keys_copy:
            async with self.lock_store.lock(key):
                try:
                    await self._process_one_key_anonymization(key)
                except Exception as e:
                    structlogger.error(
                        "rasa.privacy_manager.error_anonymizing_tracker",
                        sender_id=key,
                        error=str(e),
                    )
                    continue

    async def _process_one_key_deletion(self, key: str) -> None:
        """Process one tracker for deletion (call under lock)."""
        full_tracker = await self.tracker_store.retrieve_full_tracker(key)
        if full_tracker is None:
            structlogger.debug(
                "rasa.privacy_manager.no_tracker_found_for_sender_id",
                sender_id=key,
            )
            return None

        if not self._is_tracker_eligible_for_privacy_job(full_tracker, "deletion"):
            structlogger.debug(
                "rasa.privacy_manager.tracker_not_eligible_for_deletion",
                sender_id=full_tracker.sender_id,
            )
            return None

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
            return None
        try:
            tracker = DialogueStateTracker.from_events(
                sender_id=key,
                evts=events_to_be_retained,
                slots=full_tracker.slots.values(),
                user_id=full_tracker.user_id,
            )
        except (JsonPatchException, JsonPointerException) as exception:
            structlogger.error(
                "rasa.privacy_manager.error_reconstructing_tracker_after_deletion",
                sender_id=key,
                error=str(exception),
                event_info="Could not reconstruct tracker with events "
                "to be retained after deletion. "
                "To avoid data loss, the tracker will not be deleted. "
                "To reset the tracker's dialogue stack and enable "
                "deletion of sessions scheduled for deletion "
                "you can append a Restarted Event to the tracker.",
            )
            return None

        await self.tracker_store.update(tracker)
        structlogger.info(
            "rasa.privacy_manager.overwritten_tracker",
            sender_id=key,
            event_info="Deleted eligible events and saved "
            "tracker with events not scheduled "
            "for deletion yet.",
        )
        return None

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
            async with self.lock_store.lock(key):
                try:
                    await self._process_one_key_deletion(key)
                except Exception as e:
                    structlogger.error(
                        "rasa.privacy_manager.error_deleting_tracker",
                        sender_id=key,
                        error=str(e),
                    )
                    continue

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
    ) -> Tuple[List[Event], int]:
        """Return (all_events in chronological order, count of events anonymized).

        Order is preserved so that runs/sessions are not reordered (e.g. an
        earlier uneligible run stays before a later processed run).
        """
        if self.user_chat_inactivity_in_minutes is None:
            if not self._tracker_has_any_session_id(full_tracker):
                return self._get_processed_events_after_anonymization_legacy(
                    full_tracker.sender_id, list(full_tracker.events)
                )
            return self._get_processed_events_after_anonymization_event_only(
                full_tracker
            )
        return self._get_processed_events_after_anonymization_env_set(full_tracker)

    def _get_processed_events_after_anonymization_env_set(
        self, full_tracker: DialogueStateTracker
    ) -> Tuple[List[Event], int]:
        """Env-set path: time-based anonymization. Returns (all_events in order,
        count of events anonymized this run).
        """
        sessions = self._get_env_set_session_trackers(full_tracker)
        all_events_ordered: List[Event] = []
        num_processed = 0
        threshold_seconds = self._get_env_set_threshold_seconds("anonymization")
        current_time = time.time()

        for sender_id, session_events in sessions:
            if self._has_session_been_anonymized(session_events):
                structlogger.debug(
                    "rasa.privacy_manager.session_already_anonymized",
                    sender_id=sender_id,
                    session_id=session_events[-1].metadata.get(
                        METADATA_SESSION_ID, NO_SESSION_ID_KEY
                    ),
                )
                all_events_ordered.extend(session_events)
                continue
            if not session_events:
                continue
            last_ts = session_events[-1].timestamp
            last_event_timestamp = str(datetime.datetime.fromtimestamp(last_ts))
            if (current_time - last_ts) > threshold_seconds:
                structlogger.info(
                    ANONYMIZATION_LOG_KEY,
                    sender_id=sender_id,
                    session_id=session_events[-1].metadata.get(
                        METADATA_SESSION_ID, NO_SESSION_ID_KEY
                    ),
                    last_event_timestamp=last_event_timestamp,
                    triggered_by="anonymization_cron_job",
                )
                anonymized = self.process_events_from_segment(session_events)
                all_events_ordered.extend(anonymized)
                num_processed += len(anonymized)
            else:
                all_events_ordered.extend(session_events)
                structlogger.debug(
                    "rasa.privacy_manager.session_not_valid_for_anonymization",
                    sender_id=sender_id,
                    session_id=session_events[-1].metadata.get(
                        METADATA_SESSION_ID, NO_SESSION_ID_KEY
                    ),
                    last_event_timestamp=last_event_timestamp,
                )
        return all_events_ordered, num_processed

    def _get_processed_events_after_anonymization_legacy(
        self, sender_id: str, events: List[Event]
    ) -> Tuple[List[Event], int]:
        """Legacy path: no session_id; split by action_session_start or
        inactive/ended when only one; time-based with LEGACY_DEFAULT_INACTIVITY.
        Returns (all_events in order, count of events anonymized this run).
        """
        sessions = self._get_legacy_session_events(sender_id, events)
        all_events_ordered: List[Event] = []
        num_processed = 0
        threshold_seconds = self._get_legacy_threshold_seconds("anonymization")
        current_time = time.time()

        for seg_sender_id, session_events in sessions:
            if self._has_session_been_anonymized(session_events):
                all_events_ordered.extend(session_events)
                continue
            if not session_events:
                continue
            last_ts = session_events[-1].timestamp
            if (current_time - last_ts) > threshold_seconds:
                structlogger.info(
                    ANONYMIZATION_LOG_KEY,
                    sender_id=seg_sender_id,
                    session_id=session_events[-1].metadata.get(
                        METADATA_SESSION_ID, NO_SESSION_ID_KEY
                    ),
                    last_event_timestamp=str(datetime.datetime.fromtimestamp(last_ts)),
                    triggered_by="anonymization_cron_job",
                )
                anonymized = self.process_events_from_segment(session_events)
                all_events_ordered.extend(anonymized)
                num_processed += len(anonymized)
            else:
                all_events_ordered.extend(session_events)
        return all_events_ordered, num_processed

    def _event_only_anonymize_run_with_inactive_segments(
        self,
        run_events: List[Event],
        session_key: str,
        sender_id: str,
        min_after_seconds: float,
        current_time: float,
    ) -> Tuple[List[Event], int]:
        """Anonymize segments of a run that end with inactive/ended and are past grace.

        Returns (events for this run in order, count of events anonymized).
        """
        events_for_this_run: List[Event] = []
        num_anonymized_this_run = 0
        for segment, past_grace in self._iter_segments_with_grace_status(
            run_events, min_after_seconds, current_time
        ):
            if past_grace:
                structlogger.info(
                    ANONYMIZATION_LOG_KEY,
                    sender_id=sender_id,
                    session_id=session_key,
                    last_event_timestamp=str(
                        datetime.datetime.fromtimestamp(segment[-1].timestamp)
                    ),
                    triggered_by="anonymization_cron_job",
                )
                anonymized = self.process_events_from_segment(segment)
                events_for_this_run.extend(anonymized)
                num_anonymized_this_run += len(anonymized)
            else:
                events_for_this_run.extend(segment)
        return events_for_this_run, num_anonymized_this_run

    def _event_only_process_one_run(
        self,
        session_key: str,
        run_events: List[Event],
        later_processed: bool,
        min_after_seconds: float,
        current_time: float,
        sender_id: str,
    ) -> Tuple[List[Event], int, bool]:
        """Process one run in the event-only path (reverse chronological order).

        Returns (events for tracker, num anonymized, new later_processed).
        """
        if self._has_session_been_anonymized(run_events):
            return (
                list(run_events),
                0,
                True,  # so earlier runs that lack inactive get anonymized too
            )
        lacks_inactive = not self._session_events_contain_inactive_or_ended(run_events)
        if lacks_inactive and later_processed:
            anonymized = self.process_events_from_segment(run_events)
            return anonymized, len(anonymized), later_processed
        # Legacy sub-sessions (NO_SESSION_ID_KEY) typically have no
        # ConversationInactive; allow time-based eligibility.
        if lacks_inactive and session_key == NO_SESSION_ID_KEY and run_events:
            legacy_threshold = self._get_legacy_threshold_seconds("anonymization")
            if (current_time - run_events[-1].timestamp) > legacy_threshold:
                anonymized = self.process_events_from_segment(run_events)
                return anonymized, len(anonymized), True
        if lacks_inactive:
            return list(run_events), 0, later_processed
        events_for_run, num_anonymized = (
            self._event_only_anonymize_run_with_inactive_segments(
                run_events, session_key, sender_id, min_after_seconds, current_time
            )
        )
        return (
            events_for_run,
            num_anonymized,
            later_processed or (num_anonymized > 0),
        )

    def _event_only_process_legacy_run(
        self,
        run_events: List[Event],
        sender_id: str,
        later_processed: bool,
        min_after_seconds: float,
        current_time: float,
    ) -> Tuple[List[Event], int, bool]:
        """Process the NO_SESSION_ID_KEY run (hybrid legacy prefix).

        Expands run_events via _get_legacy_session_events and processes each
        sub-session individually so we don't anonymize the entire prefix when
        only some sub-sessions are eligible. Returns (events, num_anonymized,
        new_later_processed).
        """
        legacy_sessions = self._get_legacy_session_events(sender_id, run_events)
        sub_outputs: List[Tuple[List[Event], int]] = []
        for _sid, sub_events in reversed(legacy_sessions):
            events_out, num_anonymized, later_processed = (
                self._event_only_process_one_run(
                    NO_SESSION_ID_KEY,
                    sub_events,
                    later_processed,
                    min_after_seconds,
                    current_time,
                    sender_id,
                )
            )
            sub_outputs.append((events_out, num_anonymized))
        # Reassemble in chronological order (we processed reverse order)
        events_out = []
        num_anonymized = 0
        for evts, n in reversed(sub_outputs):
            events_out.extend(evts)
            num_anonymized += n
        return events_out, num_anonymized, later_processed

    def _get_processed_events_after_anonymization_event_only(
        self, full_tracker: DialogueStateTracker
    ) -> Tuple[List[Event], int]:
        """Event-only path (new/hybrid trackers): process by runs to preserve
        chronological order; anonymize segments with ConversationInactive or
        SessionEnded only after min_after_session_end has elapsed.
        Hybrid trackers: the legacy prefix (NO_SESSION_ID_KEY run) is expanded
        via _get_legacy_session_events and each sub-session is evaluated
        individually (matching env-set path behavior).
        If a later run is anonymized, earlier runs that lack inactive are also
        anonymized (so we don't leave old PII before an anonymized run).
        Returns (all_events in run order, count of events anonymized this run).
        """
        if self.config is None or self.config.tracker_store_settings is None:
            raise RasaException(
                "Privacy config and tracker store settings are required "
                "for event-only anonymization."
            )
        runs = self._group_events_into_runs(list(full_tracker.events))
        policy = self.config.tracker_store_settings.anonymization_policy
        min_after_seconds = policy.min_after_session_end * 60  # type: ignore[union-attr]
        current_time = time.time()

        run_outputs: List[Tuple[List[Event], int]] = []
        later_processed = False
        for session_key, run_events in reversed(runs):
            if not run_events:
                continue
            if session_key == NO_SESSION_ID_KEY:
                events_out, num_anonymized, later_processed = (
                    self._event_only_process_legacy_run(
                        run_events,
                        full_tracker.sender_id,
                        later_processed,
                        min_after_seconds,
                        current_time,
                    )
                )
                run_outputs.append((events_out, num_anonymized))
            else:
                events_out, num_anonymized, later_processed = (
                    self._event_only_process_one_run(
                        session_key,
                        run_events,
                        later_processed,
                        min_after_seconds,
                        current_time,
                        full_tracker.sender_id,
                    )
                )
                run_outputs.append((events_out, num_anonymized))

        all_events_ordered = []
        num_processed = 0
        for run_events_out, num_anonymized in reversed(run_outputs):
            all_events_ordered.extend(run_events_out)
            num_processed += num_anonymized
        return all_events_ordered, num_processed

    def _get_events_to_be_retained_after_deletion(
        self, full_tracker: DialogueStateTracker
    ) -> List[Event]:
        """Get the events to be retained after deletion."""
        if self.user_chat_inactivity_in_minutes is not None:
            return self._get_events_to_be_retained_after_deletion_env_set(full_tracker)

        # Event-only path or legacy path without env var:
        # if tracker is eligible for deletion, we delete all events
        # except when the tracker is not terminated,
        # in which case we retain all events until the tracker is terminated.
        # This is to prevent deleting events that are still relevant
        # for an active conversation.
        if full_tracker.terminated:
            return []
        else:
            structlogger.warning(
                "rasa.privacy_manager.deletion_eligibility_without_env_warning",
                sender_id=full_tracker.sender_id,
                event_info="Tracker is eligible for deletion but will not be deleted "
                "because tracker is not terminated. "
                "To terminate a tracker, you can append a "
                "SessionEnded Event to the tracker. ",
            )
            return list(full_tracker.events)

    def _get_events_to_be_retained_after_deletion_env_set(
        self, full_tracker: DialogueStateTracker
    ) -> List[Event]:
        """Env-set path: time-based retention; session splitting respects
        new/hybrid (group by session_id; legacy segment split). Uses events only.
        """
        sessions = self._get_env_set_session_trackers(full_tracker)
        events_to_be_retained: List[Event] = []
        threshold_seconds = self._get_env_set_threshold_seconds("deletion")
        current_time = time.time()

        for _sender_id, session_events in sessions:
            if not session_events:
                continue
            if (current_time - session_events[-1].timestamp) <= threshold_seconds:
                events_to_be_retained.extend(session_events)
            else:
                structlogger.info(
                    DELETION_LOG_KEY,
                    sender_id=full_tracker.sender_id,
                    last_event_timestamp=str(
                        datetime.datetime.fromtimestamp(session_events[-1].timestamp)
                    ),
                    triggered_by="deletion_cron_job",
                )
        return events_to_be_retained


def get_next_fire_time(
    privacy_policy: PrivacyPolicy,
) -> Optional[datetime.datetime]:
    """Get the next fire time for the privacy policy."""
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return privacy_policy.cron.get_next_fire_time(None, now=now)
