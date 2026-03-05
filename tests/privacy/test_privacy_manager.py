import datetime
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, AsyncGenerator, ContextManager, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import freezegun
import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from jsonpatch import JsonPatchException
from jsonpointer import JsonPointerException
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.core.lock_store import InMemoryLockStore, LockStore
from rasa.core.tracker_stores.tracker_store import (
    InMemoryTrackerStore,
)
from rasa.privacy.constants import (
    NO_SESSION_ID_KEY,
    USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME,
)
from rasa.privacy.privacy_config import PrivacyConfig
from rasa.privacy.privacy_filter import PrivacyFilter
from rasa.privacy.privacy_manager import BackgroundPrivacyManager
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    ConversationInactive,
    Event,
    SessionEnded,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import METADATA_SESSION_ID
from tests.conftest import write_endpoint_config_to_yaml
from tests.utilities import filter_logs

freezegun.config.configure(extend_ignore_list=["transformers"])


@pytest.fixture
def available_endpoints_for_pii_management() -> AvailableEndpoints:
    return Configuration.initialise_endpoints(
        endpoints_path=Path("data/test_privacy/endpoints_with_valid_privacy.yml")
    ).endpoints


@pytest.fixture
def pii_domain() -> Domain:
    """Fixture to create a domain with PII slots for testing."""
    return Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
            credit_card_number:
                type: text
            age:
                type: float
        """
    )


@pytest.fixture
def pii_tracker(pii_domain: Domain) -> DialogueStateTracker:
    """Fixture to create a tracker with PII events for testing."""
    sender_id = uuid.uuid4().hex
    return DialogueStateTracker.from_events(
        sender_id,
        evts=[
            ActionExecuted("action_session_start"),
            ActionExecuted("action_listen"),
            UserUttered("I've been double-charged for my rent this month."),
            BotUttered(
                "Sorry to hear that! Can you please provide your "
                "credit card number and your age so I can look into this for you?"
            ),
            UserUttered("Sure, it's 1234-5678-9012-3456 and I'm 24 years old."),
            SlotSet("credit_card_number", "1234-5678-9012-3456"),
            SlotSet("age", 24),
            BotUttered(
                "I've recorded your dispute regarding the double charge. "
                "We will investigate this matter and get back to you shortly."
            ),
            UserUttered("Thank you!"),
            BotUttered(
                "There's been an update on your issue. We have refunded the "
                "double charge to your credit card number 1234-5678-9012-3456."
            ),
        ],
        slots=pii_domain.slots,
    )


@pytest.fixture
async def privacy_manager(
    available_endpoints_for_pii_management: AvailableEndpoints,
    lock_store: InMemoryLockStore,
    monkeypatch: MonkeyPatch,
) -> AsyncGenerator[
    BackgroundPrivacyManager,
    None,
]:
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        available_endpoints_for_pii_management,
        lock_store=lock_store,
    )
    yield privacy_manager
    privacy_manager.stop()


@pytest.fixture
async def empty_privacy_manager(
    lock_store: InMemoryLockStore,
    monkeypatch: MonkeyPatch,
) -> AsyncGenerator[
    BackgroundPrivacyManager,
    None,
]:
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints(),
        lock_store=lock_store,
    )
    yield privacy_manager
    privacy_manager.stop()


@pytest.fixture
def unanonymized_tracker(pii_domain: Domain) -> DialogueStateTracker:
    """Fixture to create a tracker with events that should not be anonymized."""
    sender_id = uuid.uuid4().hex
    return DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted("action_session_start", timestamp=1717589977.902456),
            ActionExecuted("action_listen", timestamp=1717589977.902456),
            UserUttered(
                "I want to report my credit card 1234-5678-9012-3456 as lost.",
                timestamp=1717589977.902456,
            ),
            SlotSet(
                "credit_card_number", "1234-5678-9012-3456", timestamp=1717589977.902456
            ),
            BotUttered(
                "Your credit card 1234-5678-9012-3456 has been reported as lost.",
                timestamp=1717589977.902456,
            ),
        ],
        slots=pii_domain.slots,
    )


@pytest.fixture
def lock_store() -> InMemoryLockStore:
    """Provide an in-memory lock store for BackgroundPrivacyManager (required)."""
    return InMemoryLockStore()


@pytest.fixture
def mock_run_tracker_store_anonymization() -> Generator[AsyncMock, None, None]:
    """Fixture to mock the run_tracker_store_anonymization method."""
    mock = AsyncMock()
    yield mock
    mock.reset_mock()


@pytest.fixture
def mock_run_tracker_store_deletion() -> Generator[AsyncMock, None, None]:
    """Fixture to mock the run_tracker_store_deletion method."""
    mock = AsyncMock()
    yield mock
    mock.reset_mock()


@pytest.fixture
def deletion_privacy_config_data() -> Dict[str, Any]:
    """Fixture to provide deletion privacy config data."""
    return {
        "privacy": {
            "tracker_store_settings": {
                "deletion": {
                    "min_after_session_end": 1,
                    "cron": "30 0 * * *",
                },
            },
            "rules": [
                {"slot": "credit_card_number", "anonymization": {"type": "mask"}}
            ],
        },
    }


@pytest.fixture
def anonymization_privacy_config_data() -> Dict[str, Any]:
    """Fixture to provide anonymization privacy config data."""
    return {
        "privacy": {
            "tracker_store_settings": {
                "anonymization": {
                    "min_after_session_end": 1,
                    "cron": "30 1 * * *",
                },
            },
            "rules": [
                {"slot": "credit_card_number", "anonymization": {"type": "mask"}},
                {"slot": "age", "anonymization": {"type": "mask"}},
            ],
        },
    }


@pytest.fixture
def privacy_config_data() -> Dict[str, Any]:
    """Fixture to provide a complete privacy config data."""
    return {
        "privacy": {
            "tracker_store_settings": {
                "deletion": {
                    "min_after_session_end": 2,
                    "cron": "00 12 * * *",
                },
                "anonymization": {
                    "min_after_session_end": 1,
                    "cron": "30 11 * * *",
                },
            },
            "rules": [
                {"slot": "credit_card_number", "anonymization": {"type": "mask"}}
            ],
        },
    }


@pytest.fixture(autouse=True)
def monkeypatch_gliner(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gliner", MagicMock())
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )


async def test_create_background_privacy_manager(
    privacy_manager: BackgroundPrivacyManager,
) -> None:
    assert privacy_manager is not None
    assert isinstance(privacy_manager, BackgroundPrivacyManager)

    assert privacy_manager.tracker_store.event_broker is None
    assert isinstance(privacy_manager.tracker_store, InMemoryTrackerStore)

    assert len(privacy_manager.event_brokers) == 1
    broker = privacy_manager.event_brokers[0]
    assert isinstance(broker, KafkaEventBroker)
    assert broker.topic == "anonymization"
    assert broker.stream_pii is False

    assert privacy_manager.background_scheduler is not None
    assert isinstance(privacy_manager.background_scheduler, BackgroundScheduler)

    deletion_cron_job = privacy_manager.background_scheduler.get_job(
        job_id="deletion_cron_job"
    )
    assert deletion_cron_job is not None
    assert isinstance(deletion_cron_job.trigger, CronTrigger)
    assert deletion_cron_job.max_instances == 1

    anonymization_cron_job = privacy_manager.background_scheduler.get_job(
        job_id="anonymization_cron_job"
    )
    assert anonymization_cron_job is not None
    assert isinstance(anonymization_cron_job.trigger, CronTrigger)
    assert anonymization_cron_job.max_instances == 1

    assert isinstance(privacy_manager.config, PrivacyConfig)
    assert isinstance(privacy_manager.privacy_filter, PrivacyFilter)
    assert (
        "credit_card_number"
        in privacy_manager.privacy_filter.anonymization_rules.keys()
    )
    # Env var not set in fixture -> no default 30
    assert privacy_manager.user_chat_inactivity_in_minutes is None


async def test_create_background_privacy_manager_without_lock_store(
    monkeypatch: MonkeyPatch,
    available_endpoints_for_pii_management: AvailableEndpoints,
) -> None:
    error_message = (
        "LockStore is required for BackgroundPrivacyManager. "
        "Pass lock_store when creating the manager."
    )
    with pytest.raises(RasaException, match=error_message):
        await BackgroundPrivacyManager.create_instance(
            available_endpoints_for_pii_management,
            lock_store=None,
        )


@pytest.mark.asyncio
async def test_init_user_chat_inactivity_env_unset_is_none(
    available_endpoints_for_pii_management: AvailableEndpoints,
    monkeypatch: MonkeyPatch,
    lock_store: LockStore,
) -> None:
    """When USER_CHAT_INACTIVITY_IN_MINUTES is unset, attribute is None (no 30 min default)."""  # noqa: E501
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    privacy_manager = await BackgroundPrivacyManager.create_instance(
        available_endpoints_for_pii_management,
        lock_store=lock_store,
    )
    assert privacy_manager.user_chat_inactivity_in_minutes is None
    privacy_manager.stop()


@pytest.mark.asyncio
async def test_init_user_chat_inactivity_env_set_parsed_and_deprecation_warning(
    available_endpoints_for_pii_management: AvailableEndpoints,
    monkeypatch: MonkeyPatch,
    lock_store: LockStore,
) -> None:
    """When USER_CHAT_INACTIVITY_IN_MINUTES is set, value is parsed and deprecation warning raised."""  # noqa: E501
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "45")

    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            available_endpoints_for_pii_management,
            lock_store=lock_store,
        )
    assert privacy_manager.user_chat_inactivity_in_minutes == 45
    privacy_manager.stop()


def test_tracker_has_any_session_id(pii_domain: Domain) -> None:
    """_tracker_has_any_session_id is True when any event has session_id in metadata."""
    sender_id = "sender"
    tracker_no_session = DialogueStateTracker.from_events(
        sender_id,
        [ActionExecuted("action_session_start"), UserUttered("hi")],
        slots=pii_domain.slots,
    )
    assert not BackgroundPrivacyManager._tracker_has_any_session_id(tracker_no_session)
    evts = [
        ActionExecuted("action_session_start", metadata={METADATA_SESSION_ID: "s1"}),
        UserUttered("hi"),
    ]
    tracker_with_session = DialogueStateTracker.from_events(
        sender_id, evts, slots=pii_domain.slots
    )
    assert BackgroundPrivacyManager._tracker_has_any_session_id(tracker_with_session)


def test_group_events_by_session_id() -> None:
    """_group_events_by_session_id groups events by session_id in metadata."""
    e1 = ActionExecuted("action_session_start")
    e2 = UserUttered("hi", metadata={METADATA_SESSION_ID: "s1"})
    e3 = BotUttered("ok", metadata={METADATA_SESSION_ID: "s1"})
    e4 = UserUttered("bye", metadata={METADATA_SESSION_ID: "s2"})
    e5 = ActionExecuted("action_listen")
    events = [e1, e2, e3, e4, e5]
    grouped = BackgroundPrivacyManager._group_events_by_session_id(events)
    assert set(grouped.keys()) == {NO_SESSION_ID_KEY, "s1", "s2"}
    assert grouped[NO_SESSION_ID_KEY] == [e1, e5]
    assert grouped["s1"] == [e2, e3]
    assert grouped["s2"] == [e4]


def test_group_events_by_session_id_empty() -> None:
    """_group_events_by_session_id returns empty dict for empty list."""
    assert BackgroundPrivacyManager._group_events_by_session_id([]) == {}


def test_group_events_into_runs_legacy_prefix_then_session_id() -> None:
    """_group_events_into_runs: hybrid = legacy prefix then session_id runs."""
    e0 = ActionExecuted("action_session_start")
    e1 = UserUttered("legacy", timestamp=1)
    e2 = UserUttered("new", timestamp=2, metadata={METADATA_SESSION_ID: "s1"})
    e3 = BotUttered("ok", timestamp=3, metadata={METADATA_SESSION_ID: "s1"})
    events = [e0, e1, e2, e3]
    runs = BackgroundPrivacyManager._group_events_into_runs(events)
    assert len(runs) == 2
    assert runs[0] == (NO_SESSION_ID_KEY, [e0, e1])
    assert runs[1] == ("s1", [e2, e3])
    reassembled = [evt for _, run_evts in runs for evt in run_evts]
    assert reassembled == events


def test_group_events_into_runs_remainder_runs_preserve_order() -> None:
    """_group_events_into_runs: remainder after legacy prefix split into runs by key."""
    e0 = ActionExecuted("action_session_start")
    e1 = UserUttered("hi", metadata={METADATA_SESSION_ID: "s1"})
    e2 = ActionExecuted("action_listen")
    e3 = BotUttered("ok", metadata={METADATA_SESSION_ID: "s1"})
    events = [e0, e1, e2, e3]
    runs = BackgroundPrivacyManager._group_events_into_runs(events)
    assert len(runs) == 4
    assert runs[0] == (NO_SESSION_ID_KEY, [e0])
    assert runs[1] == ("s1", [e1])
    assert runs[2] == (NO_SESSION_ID_KEY, [e2])
    assert runs[3] == ("s1", [e3])
    reassembled = [evt for _, run_evts in runs for evt in run_evts]
    assert reassembled == events


def test_session_events_contain_inactive_or_ended() -> None:
    """_session_events_contain_inactive_or_ended detects ConversationInactive and SessionEnded."""  # noqa: E501
    assert not BackgroundPrivacyManager._session_events_contain_inactive_or_ended(
        [ActionExecuted("action_listen"), UserUttered("hi")]
    )
    assert BackgroundPrivacyManager._session_events_contain_inactive_or_ended(
        [UserUttered("hi"), ConversationInactive()]
    )
    assert BackgroundPrivacyManager._session_events_contain_inactive_or_ended(
        [UserUttered("bye"), SessionEnded()]
    )
    assert BackgroundPrivacyManager._session_events_contain_inactive_or_ended(
        [SessionEnded()]
    )


def test_split_events_by_inactive_or_ended() -> None:
    """_split_events_by_inactive_or_ended splits at ConversationInactive/
    SessionEnded, including the boundary event."""
    e1 = ActionExecuted("action_listen")
    e2 = UserUttered("hi")
    e3 = ConversationInactive()
    e4 = UserUttered("bye")
    e5 = SessionEnded()
    e6 = ActionExecuted("action_listen")
    segments = BackgroundPrivacyManager._split_events_by_inactive_or_ended(
        [e1, e2, e3, e4, e5, e6]
    )
    assert len(segments) == 3
    assert segments[0] == [e1, e2, e3]
    assert segments[1] == [e4, e5]
    assert segments[2] == [e6]
    assert BackgroundPrivacyManager._split_events_by_inactive_or_ended([]) == []


@pytest.mark.asyncio
async def test_get_env_set_session_trackers_legacy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set: legacy tracker (no session_id) yields same as
    _get_legacy_session_events."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    tracker = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted("action_session_start", timestamp=0),
            UserUttered("old", timestamp=100),
            ActionExecuted("action_session_start", timestamp=200),
            UserUttered("newer", timestamp=201),
        ],
        slots=pii_domain.slots,
    )
    assert not manager._tracker_has_any_session_id(tracker)
    sessions = manager._get_env_set_session_trackers(tracker)
    legacy_sessions = manager._get_legacy_session_events(
        tracker.sender_id, list(tracker.events)
    )
    assert len(sessions) == len(legacy_sessions) == 2
    for (sid, evts), (leg_sid, leg_evts) in zip(sessions, legacy_sessions):
        assert sid == leg_sid
        assert len(evts) == len(leg_evts)
    manager.stop()


@pytest.mark.asyncio
async def test_get_env_set_session_trackers_new(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set: new tracker (events with session_id) yields one session
    per session_id."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    evts = [
        ActionExecuted(
            "action_session_start", timestamp=0, metadata={METADATA_SESSION_ID: "s1"}
        ),
        UserUttered("hi", timestamp=1, metadata={METADATA_SESSION_ID: "s1"}),
        ActionExecuted(
            "action_session_start", timestamp=100, metadata={METADATA_SESSION_ID: "s2"}
        ),
        UserUttered("bye", timestamp=101, metadata={METADATA_SESSION_ID: "s2"}),
    ]
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    assert manager._tracker_has_any_session_id(tracker)
    sessions = manager._get_env_set_session_trackers(tracker)
    assert len(sessions) == 2
    # Sessions are (sender_id, events) tuples
    assert sessions[0][0] == "sender"
    assert sessions[1][0] == "sender"
    assert len(sessions[0][1]) == 2
    assert len(sessions[1][1]) == 2
    manager.stop()


@pytest.mark.asyncio
async def test_get_env_set_session_trackers_hybrid(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set: hybrid yields legacy segment (split by action_session_start)
    plus one session per session_id."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    # Legacy segment: two action_session_start -> 2 sub-sessions; then s1.
    evts = [
        ActionExecuted("action_session_start", timestamp=0),
        UserUttered("legacy1", timestamp=1),
        ActionExecuted("action_session_start", timestamp=10),
        UserUttered("legacy2", timestamp=11),
        ActionExecuted(
            "action_session_start", timestamp=20, metadata={METADATA_SESSION_ID: "s1"}
        ),
        UserUttered("new", timestamp=21, metadata={METADATA_SESSION_ID: "s1"}),
    ]
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    assert manager._tracker_has_any_session_id(tracker)
    sessions = manager._get_env_set_session_trackers(tracker)
    # NO_SESSION_ID_KEY segment has 4 events split into 2 legacy sessions; plus 1 for s1
    assert len(sessions) == 3
    manager.stop()


@pytest.mark.asyncio
async def test_is_tracker_eligible_event_based_terminated(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Tracker with inactive or terminated is eligible (event-based)."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    sender_id = "test-sender"
    # Tracker ending with SessionEnded -> terminated=True
    tracker_ended = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted("action_session_start"),
            UserUttered("bye"),
            SessionEnded(),
        ],
        slots=pii_domain.slots,
    )
    assert tracker_ended.terminated is True
    assert manager._is_tracker_eligible_for_privacy_job(tracker_ended, "anonymization")
    assert manager._is_tracker_eligible_for_privacy_job(tracker_ended, "deletion")
    manager.stop()


@pytest.mark.asyncio
async def test_is_tracker_eligible_env_set_time_based(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Tracker with env var set and last event older than threshold is eligible."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    # Last event at timestamp 0;
    # threshold = 0*60 + 1*60 = 60s (anonymization min_after=1)
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        tracker = DialogueStateTracker.from_events(
            "sender",
            [
                ActionExecuted("action_session_start", timestamp=0),
                UserUttered("hi", timestamp=0),
            ],
            slots=pii_domain.slots,
        )
    # current_time - 0 > 60
    assert manager._is_tracker_eligible_for_privacy_job(tracker, "anonymization")
    manager.stop()


@pytest.mark.asyncio
async def test_is_tracker_eligible_legacy_no_session_id_old_events(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Legacy tracker (no session_id on events) with old last event is eligible."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Tracker with no session_id in any event, last event long ago
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        tracker = DialogueStateTracker.from_events(
            "sender",
            [
                ActionExecuted("action_session_start", timestamp=0),
                UserUttered("hi", timestamp=0),
            ],
            slots=pii_domain.slots,
        )
    assert not manager._tracker_has_any_session_id(tracker)
    # Threshold = 30*60 + 1*60 = 1860s; current_time - 0 >> 1860
    assert manager._is_tracker_eligible_for_privacy_job(tracker, "anonymization")
    manager.stop()


@pytest.mark.asyncio
async def test_is_tracker_eligible_env_unset_has_session_id_not_inactive_not_eligible(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """When env unset, tracker with session_id but not inactive/terminated is not eligible."""  # noqa: E501
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Tracker with session_id in metadata, active (no ConversationInactive/SessionEnded)
    evts_with_meta = [
        ActionExecuted(
            "action_session_start", timestamp=1, metadata={METADATA_SESSION_ID: "s1"}
        ),
        UserUttered("hi", timestamp=2, metadata={METADATA_SESSION_ID: "s1"}),
    ]
    tracker = DialogueStateTracker.from_events(
        "sender", evts_with_meta, slots=pii_domain.slots
    )
    assert manager._tracker_has_any_session_id(tracker)
    assert not tracker.inactive and not tracker.terminated
    # Last event at 2; threshold for legacy not used (has session_id).
    # So only event-based -> False
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        assert not manager._is_tracker_eligible_for_privacy_job(
            tracker, "anonymization"
        )
    manager.stop()


@pytest.mark.asyncio
async def test_tracker_has_multiple_sessions(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """_tracker_has_multiple_sessions is True when tracker splits into >1 session."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Single session_id -> False
    single = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted(
                "action_session_start",
                timestamp=0,
                metadata={METADATA_SESSION_ID: "s1"},
            ),
            UserUttered("hi", timestamp=1, metadata={METADATA_SESSION_ID: "s1"}),
        ],
        slots=pii_domain.slots,
    )
    assert not manager._tracker_has_multiple_sessions(single)
    # Two session_ids -> True
    two_sessions = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted(
                "action_session_start",
                timestamp=0,
                metadata={METADATA_SESSION_ID: "s1"},
            ),
            UserUttered("a", timestamp=1, metadata={METADATA_SESSION_ID: "s1"}),
            ActionExecuted(
                "action_session_start",
                timestamp=100,
                metadata={METADATA_SESSION_ID: "s2"},
            ),
            UserUttered("b", timestamp=101, metadata={METADATA_SESSION_ID: "s2"}),
        ],
        slots=pii_domain.slots,
    )
    assert manager._tracker_has_multiple_sessions(two_sessions)
    # Legacy: two segments (action_session_start split) -> True
    legacy_two = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted("action_session_start", timestamp=0),
            UserUttered("old", timestamp=1),
            ActionExecuted("action_session_start", timestamp=100),
            UserUttered("recent", timestamp=101),
        ],
        slots=pii_domain.slots,
    )
    assert not legacy_two.events[0].metadata.get(METADATA_SESSION_ID)
    assert manager._tracker_has_multiple_sessions(legacy_two)
    manager.stop()


@pytest.mark.asyncio
async def test_deletion_legacy_multi_session_not_terminated_not_eligible_when_env_unset(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """With env unset, legacy tracker with multiple sessions but not terminated
    must not be eligible for deletion to avoid no-op update and lock every cron."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Legacy, multiple sessions (two action_session_start), not terminated
    legacy_multi = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted("action_session_start", timestamp=0),
            UserUttered("old", timestamp=1),
            ActionExecuted("action_session_start", timestamp=100),
            UserUttered("recent", timestamp=101),
        ],
        slots=pii_domain.slots,
    )
    assert not manager._tracker_has_any_session_id(legacy_multi)
    assert manager._tracker_has_multiple_sessions(legacy_multi)
    assert not legacy_multi.terminated
    # Without fix: multi-session shortcut made this True -> no-op update every cron
    assert not manager._is_tracker_eligible_for_privacy_job(legacy_multi, "deletion")
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_legacy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env unset + no session_id: legacy path anonymizes only old sessions
    (by action_session_start + 30 min)."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Two sessions (split by action_session_start). First old (ts 0), second recent.
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        now = time.time()
    # Legacy threshold = 30*60 + 1*60 = 1860s. First session old; second recent.
    tracker = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted("action_session_start", timestamp=0),
            UserUttered("My card is 1234-5678-9012-3456", timestamp=100),
            SlotSet("credit_card_number", "1234-5678-9012-3456", timestamp=101),
            ActionExecuted("action_session_start", timestamp=now - 100),
            UserUttered("recent session", timestamp=now - 10),
        ],
        slots=pii_domain.slots,
    )
    assert not manager._tracker_has_any_session_id(tracker)
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        all_events, num_processed = manager._get_processed_events_after_anonymization(
            tracker
        )
    # Old session anonymized; recent session uneligible; order preserved.
    assert num_processed == 3
    assert len(all_events) == 5
    # Chronological order: first session then second (not reordered).
    user_texts = [e.text for e in all_events if isinstance(e, UserUttered)]
    assert user_texts.index("recent session") > user_texts.index(
        "My card is [CREDIT_CARD_NUMBER]"
    )
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_event_only(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    pii_tracker: DialogueStateTracker,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env unset + session_id: event-only path anonymizes sessions that
    contain ConversationInactive or SessionEnded only after
    min_after_session_end has elapsed."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # privacy_config_data has min_after_session_end: 2 min; use last event >2 min ago.
    base = time.time()
    evts = list(pii_tracker.events)
    for e in evts:
        e.metadata[METADATA_SESSION_ID] = "s1"
    evts.append(
        ConversationInactive(
            timestamp=base - 180,
            metadata={METADATA_SESSION_ID: "s1"},
        )
    )
    tracker = DialogueStateTracker.from_events(
        pii_tracker.sender_id, evts, slots=pii_domain.slots
    )
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    assert num_processed == len(evts)
    assert len(all_events) == len(evts)
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_event_only_grace_period(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    pii_tracker: DialogueStateTracker,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Event-only anonymization does not anonymize sessions with inactive/ended
    until min_after_session_end has elapsed (same grace period as deletion)."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # privacy_config_data has min_after_session_end: 2 min; last event 30s ago.
    base = time.time()
    evts = list(pii_tracker.events)
    for e in evts:
        e.metadata[METADATA_SESSION_ID] = "s1"
    evts.append(
        ConversationInactive(
            timestamp=base - 30,
            metadata={METADATA_SESSION_ID: "s1"},
        )
    )
    tracker = DialogueStateTracker.from_events(
        pii_tracker.sender_id, evts, slots=pii_domain.slots
    )
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    assert num_processed == 0
    assert len(all_events) == len(evts)
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_event_only_inactive_in_middle_run_last_event_recent(  # noqa: E501
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    pii_tracker: DialogueStateTracker,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Event-only: one run with ConversationInactive in the middle and recent follow-up.

    Simulates start_session_after_expiry False: the next session does not execute
    action_session_start after expiry, so the run has
    [initial..., ConversationInactive, follow-up...].
    The run's last event is recent; we must not skip the run based on that.
    Only the segment ending with ConversationInactive (past grace) is
    anonymized; the follow-up segment stays uneligible.
    """
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # min_after_session_end is 2 min for deletion / 1 min for anonymization in config.
    # Build one run (single session_id): segment1 (old, ends with ConversationInactive)
    # then segment2 (recent follow-up). Run's last event = recent → old run-level
    # check would have skipped the whole run; we must anonymize segment1 only.
    base = time.time()
    evts = list(pii_tracker.events)
    for e in evts:
        e.metadata[METADATA_SESSION_ID] = "s1"
    # Segment 1 ends with ConversationInactive, well past grace (3 min ago).
    evts.append(
        ConversationInactive(
            timestamp=base - 180,
            metadata={METADATA_SESSION_ID: "s1"},
        )
    )
    # Segment 2: follow-up events (recent); run's last event is 10s ago.
    evts.append(
        UserUttered(
            "I want to make another payment.",
            timestamp=base - 10,
            metadata={METADATA_SESSION_ID: "s2"},
        )
    )
    evts.append(
        BotUttered(
            "What is your full name?",
            timestamp=base - 9,
            metadata={METADATA_SESSION_ID: "s2"},
        )
    )
    tracker = DialogueStateTracker.from_events(
        pii_tracker.sender_id, evts, slots=pii_domain.slots
    )
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    # First segment (s1, past grace) anonymized; follow-up (s2) retained.
    assert num_processed == len(list(pii_tracker.events)) + 1
    assert len(all_events) == len(tracker.events)
    # Order preserved: s1 segment (anonymized) before s2 segment.
    s1_events = [e for e in all_events if e.metadata.get(METADATA_SESSION_ID) == "s1"]
    s2_events = [e for e in all_events if e.metadata.get(METADATA_SESSION_ID) == "s2"]
    if s1_events and s2_events:
        assert all_events.index(s1_events[-1]) < all_events.index(s2_events[0])
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_event_only_run_without_inactive(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Event-only: run1 has no ConversationInactive (e.g. user restarted before
    timeout), run2 has ConversationInactive past grace. Run1 is eligible because
    run2 is processed. Chronological order must be preserved (run1 before run2).
    """
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    base = time.time()
    # Run 1 (s1): no ConversationInactive (e.g. user restarted before timeout).
    run1 = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 500,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered(
            "first run message",
            timestamp=base - 499,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
    ]
    # Run 2 (s2): has ConversationInactive, past grace (3 min ago).
    run2 = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 200,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        UserUttered(
            "second run message",
            timestamp=base - 199,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        ConversationInactive(timestamp=base - 180),  # no session_id (processor)
    ]
    evts = run1 + run2
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    # Run2 is anonymized (has inactive past grace). Run1 lacks inactive but is
    # before a processed run, so it is also anonymized (reverse-pass rule).
    assert num_processed >= len(evts)
    assert len(all_events) == len(evts)
    # Chronological order preserved: first UserUttered is run1, second is run2.
    user_events = [e for e in all_events if isinstance(e, UserUttered)]
    assert len(user_events) == 2
    assert all_events.index(user_events[0]) < all_events.index(user_events[1])
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_event_only_three_runs_order(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Event-only with 3 runs: output must stay in chronological order.

    Run1 no inactive (anonymized because run2 is processed), run2 has inactive
    past grace (processed), run3 no inactive (uneligible). Order must be
    run1, run2, run3.
    """
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    base = time.time()
    run1 = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 600,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered(
            "run one",
            timestamp=base - 599,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
    ]
    run2 = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 300,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        UserUttered(
            "run two",
            timestamp=base - 299,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        ConversationInactive(timestamp=base - 180),
    ]
    run3 = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 50,
            metadata={METADATA_SESSION_ID: "s3"},
        ),
        UserUttered(
            "run three",
            timestamp=base - 49,
            metadata={METADATA_SESSION_ID: "s3"},
        ),
    ]
    evts = run1 + run2 + run3
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    assert num_processed >= 4  # run1 + run2 anonymized
    assert len(all_events) == len(evts)
    user_texts = [e.text for e in all_events if isinstance(e, UserUttered)]
    assert len(user_texts) == 3
    # Chronological order: run one, run two, run three.
    assert user_texts.index("run two") > user_texts.index("run one")
    assert user_texts.index("run three") > user_texts.index("run two")
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_event_only_hybrid_legacy_subsession_not_eligible(  # noqa: E501
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Event-only + hybrid: legacy prefix has two sub-sessions; only the one
    old enough (time-based) is anonymized. Legacy sessions have no
    ConversationInactive. The other legacy sub-session and session_id run
    are recent so must not be anonymized."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Legacy threshold = LEGACY_DEFAULT_INACTIVITY (30 min) + min_after (1 min).
    # Legacy prefix: sub1 last event old (past threshold), sub2 recent. No
    # ConversationInactive in legacy (realistic). Use real PII to assert masking.
    base = time.time()
    pii_legacy1 = "1234-5678-9012-3456"
    pii_legacy2 = "9999-8888-7777-6666"
    pii_s1 = "1111-2222-3333-4444"
    legacy_sub1 = [
        ActionExecuted("action_session_start", timestamp=base - 2000),
        UserUttered(f"My card is {pii_legacy1}", timestamp=base - 1999),
        SlotSet("credit_card_number", pii_legacy1, timestamp=base - 1998),
    ]
    legacy_sub2 = [
        ActionExecuted("action_session_start", timestamp=base - 100),
        UserUttered(f"My card is {pii_legacy2}", timestamp=base - 10),
        SlotSet("credit_card_number", pii_legacy2, timestamp=base - 9),
    ]
    session_run = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 5,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered(
            f"My card is {pii_s1}",
            timestamp=base - 4,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        SlotSet(
            "credit_card_number",
            pii_s1,
            timestamp=base - 4,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
    ]
    evts = legacy_sub1 + legacy_sub2 + session_run
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    assert manager._tracker_has_any_session_id(tracker)
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    # Only first legacy sub-session (old) anonymized; legacy2 and s1 run (recent) not.
    assert num_processed >= len(legacy_sub1)
    user_texts = [e.text for e in all_events if isinstance(e, UserUttered)]
    assert not any(
        pii_legacy1 in t for t in user_texts
    ), "First legacy sub-session must be anonymized (PII masked)"
    assert any(
        pii_legacy2 in t for t in user_texts
    ), "Second legacy sub-session must not be anonymized (PII retained)"
    assert any(
        pii_s1 in t for t in user_texts
    ), "Session_id run must not be anonymized (PII retained)"
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_env_set_legacy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set + legacy: time-based path anonymizes only sessions older than
    env threshold (same split as legacy)."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    # Real time so manager's time.time() matches; threshold = 0*60+1*60 = 60s.
    base = time.time()
    tracker = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted("action_session_start", timestamp=base - 1000),
            UserUttered("old session", timestamp=base - 1000),
            ActionExecuted("action_session_start", timestamp=base - 100),
            UserUttered("recent session", timestamp=base - 10),
        ],
        slots=pii_domain.slots,
    )
    assert not manager._tracker_has_any_session_id(tracker)
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    assert num_processed >= 2
    assert len(all_events) == 4
    user_texts = [e.text for e in all_events if isinstance(e, UserUttered)]
    assert user_texts.index("recent session") > user_texts.index("old session")
    manager.stop()


@pytest.mark.asyncio
async def test_get_processed_events_after_anonymization_env_set_new_tracker(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set + new tracker: one session per session_id; only old
    session anonymized."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    base = time.time()
    evts = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 1000,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered("old", timestamp=base - 1000, metadata={METADATA_SESSION_ID: "s1"}),
        ActionExecuted(
            "action_session_start",
            timestamp=base - 100,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        UserUttered(
            "recent", timestamp=base - 10, metadata={METADATA_SESSION_ID: "s2"}
        ),
    ]
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    assert manager._tracker_has_any_session_id(tracker)
    all_events, num_processed = manager._get_processed_events_after_anonymization(
        tracker
    )
    assert num_processed >= 2
    assert len(all_events) == 4
    user_texts = [e.text for e in all_events if isinstance(e, UserUttered)]
    assert user_texts.index("recent") > user_texts.index("old")
    manager.stop()


@pytest.mark.parametrize(
    "tracker_events, expected_retained_events",
    [
        (
            [
                ActionExecuted("action_session_start", timestamp=1),
                UserUttered("old", timestamp=2),
                ActionExecuted("action_session_start", timestamp=3),
                UserUttered("recent", timestamp=4),
            ],
            4,  # all events retained since no SessionEnded as terminal event
        ),
        (
            [
                ActionExecuted("action_session_start", timestamp=1),
                UserUttered("old", timestamp=2),
                ActionExecuted("action_session_start", timestamp=3),
                UserUttered("recent", timestamp=4),
                ConversationInactive(timestamp=5),
            ],
            5,  # all events retained since ConversationInactive is not a terminal event
        ),
        (
            [
                ActionExecuted("action_session_start", timestamp=1),
                UserUttered("old", timestamp=2),
                ActionExecuted("action_session_start", timestamp=3),
                UserUttered("recent", timestamp=4),
                SessionEnded(timestamp=5),
            ],
            0,  # all events dropped since tracker is terminated
        ),
    ],
)
@pytest.mark.asyncio
async def test_get_events_to_be_retained_after_deletion_legacy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
    tracker_events: List[Event],
    expected_retained_events: int,
) -> None:
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )

    tracker = DialogueStateTracker.from_events(
        "sender",
        tracker_events,
        slots=pii_domain.slots,
    )
    retained = manager._get_events_to_be_retained_after_deletion(tracker)
    assert len(retained) == expected_retained_events
    manager.stop()


@pytest.mark.asyncio
async def test_get_events_to_be_retained_after_deletion_event_only(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env unset: event-only retains sessions without inactive/ended or
    within min_after_session_end."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
        lock_store=lock_store,
    )
    # Session with SessionEnded, last event old (past min_after=2 min) -> drop
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        now = time.time()
    old_ts = now - 180  # 3 minutes ago so past 2*60s threshold
    evts = [
        ActionExecuted(
            "action_session_start",
            timestamp=old_ts,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered("bye", timestamp=old_ts + 1, metadata={METADATA_SESSION_ID: "s1"}),
        SessionEnded(
            timestamp=old_ts + 2,
            metadata={METADATA_SESSION_ID: "s1"},
        ),  # last event old; same session_id so grouped
    ]
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    with freezegun.freeze_time("2020-01-01 12:00:00"):
        retained = manager._get_events_to_be_retained_after_deletion(tracker)
    assert len(retained) == 0
    manager.stop()


@pytest.mark.asyncio
async def test_get_events_to_be_retained_after_deletion_env_set_legacy(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set + legacy: time-based path retains only sessions within
    env threshold."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    base = time.time()
    tracker = DialogueStateTracker.from_events(
        "sender",
        [
            ActionExecuted("action_session_start", timestamp=base - 1000),
            UserUttered("old", timestamp=base - 1000),
            ActionExecuted("action_session_start", timestamp=base - 100),
            UserUttered("recent", timestamp=base - 10),
        ],
        slots=pii_domain.slots,
    )
    assert not manager._tracker_has_any_session_id(tracker)
    retained = manager._get_events_to_be_retained_after_deletion(tracker)
    assert len(retained) == 2  # recent session only
    manager.stop()


@pytest.mark.asyncio
async def test_get_events_to_be_retained_after_deletion_env_set_new_tracker(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """Env set + new tracker: one session per session_id; only recent
    session retained."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, privacy_config_data)
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=InMemoryTrackerStore(pii_domain),
            lock_store=lock_store,
        )
    base = time.time()
    evts = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 1000,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered("old", timestamp=base - 1000, metadata={METADATA_SESSION_ID: "s1"}),
        ActionExecuted(
            "action_session_start",
            timestamp=base - 100,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        UserUttered(
            "recent", timestamp=base - 10, metadata={METADATA_SESSION_ID: "s2"}
        ),
    ]
    tracker = DialogueStateTracker.from_events("sender", evts, slots=pii_domain.slots)
    assert manager._tracker_has_any_session_id(tracker)
    retained = manager._get_events_to_be_retained_after_deletion(tracker)
    assert len(retained) == 2  # s2 only
    manager.stop()


def test_privacy_manager_process_events_all(
    privacy_manager: BackgroundPrivacyManager,
    pii_tracker: DialogueStateTracker,
) -> None:
    anonymized_events = privacy_manager.process_events(pii_tracker, process_all=True)

    assert len(anonymized_events) == len(pii_tracker.events)

    second_user_message = anonymized_events[4]
    assert isinstance(second_user_message, UserUttered)
    assert (
        second_user_message.text
        == "Sure, it's ###############3456 and I'm [AGE] years old."
    )

    first_slot_set_message = anonymized_events[5]
    assert isinstance(first_slot_set_message, SlotSet)
    assert first_slot_set_message.key == "credit_card_number"
    assert first_slot_set_message.value == "###############3456"

    second_slot_set_message = anonymized_events[6]
    assert isinstance(second_slot_set_message, SlotSet)
    assert second_slot_set_message.key == "age"
    assert second_slot_set_message.value == "[AGE]"

    last_bot_message = anonymized_events[-1]
    assert isinstance(last_bot_message, BotUttered)
    assert last_bot_message.text == (
        "There's been an update on your issue. "
        "We have refunded the double charge to your "
        "credit card number ###############3456."
    )


def test_privacy_manager_process_events_last_turn_only(
    privacy_manager: BackgroundPrivacyManager,
    pii_tracker: DialogueStateTracker,
) -> None:
    anonymized_events = privacy_manager.process_events(pii_tracker, process_all=False)

    assert len(anonymized_events) == 2
    last_bot_message = anonymized_events[-1]
    assert isinstance(last_bot_message, BotUttered)
    assert last_bot_message.text == (
        "There's been an update on your issue. "
        "We have refunded the double charge to your "
        "credit card number ###############3456."
    )


def test_privacy_manager_stream_events_no_event_brokers(
    empty_privacy_manager: BackgroundPrivacyManager,
) -> None:
    with capture_logs() as caplog:
        empty_privacy_manager.stream_events([], "test_sender_id")
        logs = filter_logs(
            caplog,
            "rasa.privacy_manager.no_event_broker_configured",
            "debug",
        )
        assert len(logs) == 1


@freezegun.freeze_time("2024-06-05 12:19:37.902456")
def test_privacy_manager_stream_events_with_event_brokers(
    privacy_manager: BackgroundPrivacyManager,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_publish = Mock()
    monkeypatch.setattr(privacy_manager.event_brokers[0], "publish", mock_publish)

    anonymized_events = [
        SlotSet("credit_card_number", "###############3456"),
        UserUttered("Sure, it's ###############3456."),
    ]
    sender_id = uuid.uuid4().hex

    privacy_manager.stream_events(anonymized_events, sender_id)

    assert mock_publish.call_count == 2
    assert mock_publish.call_args_list == [
        call(
            {
                "sender_id": sender_id,
                "event": "slot",
                "timestamp": 1717589977.902456,
                "name": "credit_card_number",
                "value": "###############3456",
                "filled_by": None,
                "anonymized_at": None,
            }
        ),
        call(
            {
                "sender_id": sender_id,
                "event": "user",
                "timestamp": 1717589977.902456,
                "text": "Sure, it's ###############3456.",
                "parse_data": {
                    "intent": {},
                    "entities": [],
                    "text": "Sure, it's ###############3456.",
                    "message_id": None,
                    "metadata": {},
                },
                "input_channel": None,
                "message_id": None,
                "metadata": {},
                "anonymized_at": None,
            }
        ),
    ]


@freezegun.freeze_time("2024-06-05 12:19:37.902456")
def test_privacy_manager_process_events_with_no_pii(
    privacy_manager: BackgroundPrivacyManager,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_publish = Mock()
    monkeypatch.setattr(privacy_manager.event_brokers[0], "publish", mock_publish)

    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            ActionExecuted("action_session_start"),
            ActionExecuted("action_listen"),
            BotUttered("Howdy, how can I assist you today?"),
            UserUttered("Does the weather look good for a picnic?"),
        ],
    )

    # When
    privacy_manager.process(tracker, process_all=True)
    assert mock_publish.call_count == 4
    assert mock_publish.call_args_list == [
        call(
            {
                "sender_id": tracker.sender_id,
                "event": "action",
                "timestamp": 1717589977.902456,
                "name": "action_session_start",
                "policy": None,
                "confidence": None,
                "action_text": None,
                "hide_rule_turn": False,
            }
        ),
        call(
            {
                "sender_id": tracker.sender_id,
                "event": "action",
                "timestamp": 1717589977.902456,
                "name": "action_listen",
                "policy": None,
                "confidence": None,
                "action_text": None,
                "hide_rule_turn": False,
            }
        ),
        call(
            {
                "sender_id": tracker.sender_id,
                "event": "bot",
                "timestamp": 1717589977.902456,
                "text": "Howdy, how can I assist you today?",
                "data": {},
                "metadata": {},
                "anonymized_at": 1717589977.902456,
            }
        ),
        call(
            {
                "sender_id": tracker.sender_id,
                "event": "user",
                "timestamp": 1717589977.902456,
                "text": "Does the weather look good for a picnic?",
                "parse_data": {"text": ""},
                "input_channel": None,
                "message_id": None,
                "metadata": {},
                "anonymized_at": 1717589977.902456,
            }
        ),
    ]


def test_privacy_manager_get_latest_user_message_no_privacy_filter(
    empty_privacy_manager: BackgroundPrivacyManager,
) -> None:
    with capture_logs() as caplog:
        latest_message = empty_privacy_manager._get_latest_user_message(
            DialogueStateTracker.from_events("test_sender", [])
        )
        logs = filter_logs(
            caplog,
            "rasa.privacy_manager.no_privacy_rules_configured",
            "debug",
        )
        assert len(logs) == 1
        assert latest_message is None


def test_privacy_manager_get_latest_user_message_with_privacy_filter_no_events(
    privacy_manager: BackgroundPrivacyManager,
) -> None:
    with capture_logs() as caplog:
        latest_message = privacy_manager._get_latest_user_message(
            DialogueStateTracker.from_events("test_sender", [])
        )
        logs = filter_logs(
            caplog,
            "rasa.privacy_manager.no_user_message.skipping_processing",
            "debug",
        )
        assert len(logs) == 1
        assert latest_message is None


def test_privacy_manager_validate_sensitive_slots(
    privacy_manager: BackgroundPrivacyManager,
) -> None:
    exception_message = (
        "Sensitive slots defined in the privacy config "
        "do not match the slots defined in the domain. "
        "Please check the slot names."
    )
    with capture_logs() as caplog:
        with pytest.raises(RasaException, match=exception_message):
            privacy_manager.validate_sensitive_slots_in_domain(Domain.empty())

        logs = filter_logs(
            caplog,
            "privacy_config.invalid_sensitive_slot",
            log_level="error",
            log_message_parts=["Sensitive slot not found in the domain."],
        )
        assert len(logs) == 2
        assert logs[0].get("sensitive_slot") == "credit_card_number"
        assert logs[1].get("sensitive_slot") == "age"


def test_privacy_manager_validate_sensitive_slots_with_no_config(
    empty_privacy_manager: BackgroundPrivacyManager,
) -> None:
    with capture_logs() as caplog:
        empty_privacy_manager.validate_sensitive_slots_in_domain(Domain.empty())
        logs = filter_logs(
            caplog,
            "rasa.privacy_manager.no_sensitive_slots_configured",
            "debug",
        )
        assert len(logs) == 1


@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_privacy_manager_run_tracker_store_deletion(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    lock_store: LockStore,
    env_value: Any,
    expectation: ContextManager[None],
) -> None:
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
        unanonymized_tracker.update(SessionEnded())
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(
        tmp_path,
        deletion_privacy_config_data,
    )

    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=lock_store,
        )

    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain
    assert (
        len(in_memory_tracker_store.store) != 0
    ), "Tracker store reference should not be empty."

    await privacy_manager._run_tracker_store_deletion()

    tracker = await privacy_manager.tracker_store.retrieve(
        unanonymized_tracker.sender_id
    )
    assert tracker is None, "Tracker should have been deleted after inactivity."

    assert (
        len(in_memory_tracker_store.store) == 0
    ), "In-memory tracker store should be empty after deletion."

    privacy_manager.stop()


@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_privacy_manager_run_tracker_store_deletion_retained_events(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    lock_store: LockStore,
    env_value: Any,
    expectation: ContextManager[None],
) -> None:
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(
        tmp_path,
        deletion_privacy_config_data,
    )
    new_tracker_events = [
        ActionExecuted(
            "action_session_start", timestamp=datetime.datetime.now().timestamp()
        ),
        ActionExecuted("action_listen", timestamp=datetime.datetime.now().timestamp()),
        UserUttered(
            "I want to check my account balance.",
            timestamp=datetime.datetime.now().timestamp(),
        ),
    ]
    unanonymized_tracker.update_with_events(new_tracker_events)

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            lock_store=lock_store,
        )
    mock_tracker_store_delete = AsyncMock()
    monkeypatch.setattr(
        privacy_manager.tracker_store, "delete", mock_tracker_store_delete
    )

    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    await privacy_manager._run_tracker_store_deletion()

    tracker = await privacy_manager.tracker_store.retrieve(
        unanonymized_tracker.sender_id
    )
    assert tracker is not None
    assert len(tracker.events) == 3, "Tracker should retain events after inactivity."
    assert list(tracker.events) == new_tracker_events
    mock_tracker_store_delete.assert_not_called()

    privacy_manager.stop()


@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_privacy_manager_run_tracker_store_anonymization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    anonymization_privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
    env_value: Any,
    expectation: ContextManager[None],
) -> None:
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(
        tmp_path,
        anonymization_privacy_config_data,
    )

    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=lock_store,
        )

    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain
    assert (
        len(in_memory_tracker_store.store) != 0
    ), "Tracker store reference should not be empty."

    await privacy_manager._run_tracker_store_anonymization()

    tracker = await privacy_manager.tracker_store.retrieve(
        unanonymized_tracker.sender_id
    )
    assert tracker is not None
    assert len(tracker.events) == 5

    tracker_from_in_memory_store = await in_memory_tracker_store.retrieve(
        unanonymized_tracker.sender_id
    )
    assert tracker == tracker_from_in_memory_store

    assert (
        tracker.events[2].text
        == "I want to report my credit card [CREDIT_CARD_NUMBER] as lost."
    )
    assert tracker.events[3].key == "credit_card_number"
    assert tracker.events[3].value == "[CREDIT_CARD_NUMBER]"
    assert (
        tracker.events[4].text
        == "Your credit card [CREDIT_CARD_NUMBER] has been reported as lost."
    )

    privacy_manager.stop()


@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_privacy_manager_run_tracker_store_anonymization_already_anonymized(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    anonymization_privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
    env_value: Any,
    expectation: ContextManager[None],
) -> None:
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(
        tmp_path,
        anonymization_privacy_config_data,
    )

    anonymized_tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            ActionExecuted("action_session_start", timestamp=1717589977.902456),
            ActionExecuted("action_listen", timestamp=1717589977.902456),
            UserUttered(
                "I want to report my credit card [CREDIT_CARD_NUMBER] as lost.",
                timestamp=1717589977.902456,
            ),
            SlotSet(
                "credit_card_number",
                "[CREDIT_CARD_NUMBER]",
                timestamp=1717589977.902456,
            ),
            BotUttered(
                "Your credit card [CREDIT_CARD_NUMBER] has been reported as lost.",
                timestamp=1717589977.902456,
            ),
        ],
    )
    for event in reversed(list(anonymized_tracker.events)[2:]):
        event.anonymized_at = datetime.datetime.now()

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            lock_store=lock_store,
        )
    await privacy_manager.tracker_store.save(anonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    await privacy_manager._run_tracker_store_anonymization()

    new_tracker = await privacy_manager.tracker_store.retrieve(
        anonymized_tracker.sender_id
    )
    assert (
        new_tracker == anonymized_tracker
    ), "Tracker should remain unchanged as it was already anonymized."

    privacy_manager.stop()


def test_privacy_manager_process_events_with_no_latest_message(
    privacy_manager: BackgroundPrivacyManager,
) -> None:
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            ActionExecuted("action_session_start"),
            ActionExecuted("action_listen"),
            BotUttered("Hello! How can I help you today?"),
        ],
    )
    with capture_logs() as caplog:
        anonymized_events = privacy_manager.process_events(tracker, process_all=True)
        logs = filter_logs(
            caplog,
            "rasa.privacy_manager.no_user_message.skipping_processing",
            "debug",
        )
        assert len(logs) == 1
        assert len(anonymized_events) == 0


def test_privacy_manager_has_session_been_anonymized(
    empty_privacy_manager: BackgroundPrivacyManager,
) -> None:
    tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            ActionExecuted("action_session_start"),
            ActionExecuted("action_listen"),
            UserUttered("Hello!"),
        ],
    )
    assert (
        empty_privacy_manager._has_session_been_anonymized(
            list(tracker.applied_events())
        )
        is False
    )

    tracker.events[-1].anonymized_at = datetime.datetime.now()
    assert (
        empty_privacy_manager._has_session_been_anonymized(
            list(tracker.applied_events())
        )
        is True
    )


def test_privacy_manager_consumer_queue(
    privacy_manager: BackgroundPrivacyManager,
    unanonymized_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_process = Mock()
    monkeypatch.setattr(privacy_manager, "process", mock_process)
    privacy_manager.run(unanonymized_tracker)
    assert privacy_manager.tracker_queue.qsize() == 1

    time.sleep(1)
    assert privacy_manager.tracker_queue.qsize() == 0
    mock_process.assert_called_once()


@freezegun.freeze_time("2025-01-31 12:00:00")
@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_privacy_manager_run_tracker_store_background_jobs_sequentially_both(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    mock_run_tracker_store_anonymization: AsyncMock,
    mock_run_tracker_store_deletion: AsyncMock,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
    env_value: Any,
    expectation: ContextManager[None],
) -> None:
    """We freeze time to ensure that the cron jobs for both anonymization and deletion
    are triggered at the same time, allowing us to test the sequential execution
    of both jobs in the `_run_tracker_store_background_jobs_sequentially` method.
    """
    # Given
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, privacy_config_data, "endpoints_sequential.yml"
    )

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(endpoints_path=endpoints_path).endpoints,
            lock_store=lock_store,
        )
    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    monkeypatch.setattr(
        privacy_manager,
        "_run_tracker_store_anonymization",
        mock_run_tracker_store_anonymization,
    )
    monkeypatch.setattr(
        privacy_manager, "_run_tracker_store_deletion", mock_run_tracker_store_deletion
    )
    monkeypatch.setattr(
        privacy_manager,
        "previous_fire_time_deletion",
        datetime.datetime.fromisoformat("2025-01-30 19:00:00+00:00"),
    )

    # When
    await privacy_manager._run_tracker_store_background_jobs_sequentially()

    # Then
    mock_run_tracker_store_anonymization.assert_called_once()
    mock_run_tracker_store_deletion.assert_called_once()

    assert (
        privacy_manager.previous_fire_time_deletion
        == datetime.datetime.fromisoformat("2025-01-31 12:00:00+00:00")
    ), "The previous fire time for deletion should be updated to the current time."

    privacy_manager.stop()


@freezegun.freeze_time("2025-01-31 12:30:00")
@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_privacy_manager_run_tracker_store_jobs_sequentially_anonymization_only(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    mock_run_tracker_store_anonymization: AsyncMock,
    mock_run_tracker_store_deletion: AsyncMock,
    privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
    env_value: Any,
    expectation: ContextManager[None],
) -> None:
    """We freeze time to ensure that the anonymization cron job is triggered.

    The deletion cron job won't run because the current time is 12:30.
    """
    # Given
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, privacy_config_data, "endpoints_anonymization_only.yml"
    )

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(endpoints_path=endpoints_path).endpoints,
            lock_store=lock_store,
        )
    await privacy_manager.tracker_store.save(unanonymized_tracker)

    monkeypatch.setattr(
        privacy_manager,
        "_run_tracker_store_anonymization",
        mock_run_tracker_store_anonymization,
    )
    monkeypatch.setattr(
        privacy_manager, "_run_tracker_store_deletion", mock_run_tracker_store_deletion
    )

    # When
    await privacy_manager._run_tracker_store_background_jobs_sequentially()

    # Then
    mock_run_tracker_store_anonymization.assert_called_once()
    mock_run_tracker_store_deletion.assert_not_called()

    privacy_manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_run_tracker_store_anonymization_uses_lock_when_lock_store_provided(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    anonymization_privacy_config_data: Dict[str, Any],
    env_value: Optional[str],
    expectation: ContextManager[None],
) -> None:
    """When lock_store is provided, anonymization job acquires lock per key."""
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, env_value)

    write_endpoint_config_to_yaml(tmp_path, anonymization_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    mock_lock_cm = MagicMock()
    mock_lock_cm.__aenter__ = AsyncMock(return_value=None)
    mock_lock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_lock_store = MagicMock()
    mock_lock_store.lock.return_value = mock_lock_cm

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=mock_lock_store,
        )
    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    await privacy_manager._run_tracker_store_anonymization()

    mock_lock_store.lock.assert_called_once_with(unanonymized_tracker.sender_id)
    mock_lock_cm.__aenter__.assert_called_once()
    mock_lock_cm.__aexit__.assert_called_once()
    privacy_manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_run_tracker_store_deletion_uses_lock_when_lock_store_provided(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    env_value: Optional[str],
    expectation: ContextManager[None],
) -> None:
    """When lock_store is provided, deletion job acquires lock per key."""
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, env_value)

    write_endpoint_config_to_yaml(tmp_path, deletion_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    mock_lock_cm = MagicMock()
    mock_lock_cm.__aenter__ = AsyncMock(return_value=None)
    mock_lock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_lock_store = MagicMock()
    mock_lock_store.lock.return_value = mock_lock_cm

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=mock_lock_store,
        )
    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    await privacy_manager._run_tracker_store_deletion()

    mock_lock_store.lock.assert_called_once_with(unanonymized_tracker.sender_id)
    mock_lock_cm.__aenter__.assert_called_once()
    mock_lock_cm.__aexit__.assert_called_once()
    privacy_manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_lock_released_when_process_one_key_anonymization_raises(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    anonymization_privacy_config_data: Dict[str, Any],
    env_value: Optional[str],
    expectation: ContextManager[None],
) -> None:
    """When _process_one_key_anonymization raises, lock __aexit__ is still called."""
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, env_value)

    write_endpoint_config_to_yaml(tmp_path, anonymization_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    mock_lock_cm = MagicMock()
    mock_lock_cm.__aenter__ = AsyncMock(return_value=None)
    mock_lock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_lock_store = MagicMock()
    mock_lock_store.lock.return_value = mock_lock_cm

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=mock_lock_store,
        )
    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    with patch.object(
        privacy_manager,
        "_process_one_key_anonymization",
        side_effect=RuntimeError("test anonymization failure"),
    ):
        await privacy_manager._run_tracker_store_anonymization()

    mock_lock_cm.__aenter__.assert_called_once()
    mock_lock_cm.__aexit__.assert_called_once()
    privacy_manager.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "env_value, expectation",
    [
        (None, nullcontext()),
        (
            "0",
            pytest.warns(
                FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
            ),
        ),
    ],
)
async def test_lock_released_when_process_one_key_deletion_raises(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    env_value: Optional[str],
    expectation: ContextManager[None],
) -> None:
    """When _process_one_key_deletion raises, lock __aexit__ is still called."""
    if env_value is None:
        monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, env_value)

    write_endpoint_config_to_yaml(tmp_path, deletion_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    mock_lock_cm = MagicMock()
    mock_lock_cm.__aenter__ = AsyncMock(return_value=None)
    mock_lock_cm.__aexit__ = AsyncMock(return_value=None)
    mock_lock_store = MagicMock()
    mock_lock_store.lock.return_value = mock_lock_cm

    with expectation:
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=mock_lock_store,
        )
    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    with patch.object(
        privacy_manager,
        "_process_one_key_deletion",
        side_effect=RuntimeError("test deletion failure"),
    ):
        await privacy_manager._run_tracker_store_deletion()

    mock_lock_cm.__aenter__.assert_called_once()
    mock_lock_cm.__aexit__.assert_called_once()
    privacy_manager.stop()


@pytest.mark.asyncio
async def test_run_tracker_store_skips_ineligible_tracker(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    anonymization_privacy_config_data: Dict[str, Any],
    lock_store: LockStore,
) -> None:
    """When tracker is not eligible (e.g. active, env unset), no delete/save."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)
    write_endpoint_config_to_yaml(tmp_path, anonymization_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    # Tracker with session_id but no ConversationInactive/SessionEnded (active).
    active_tracker = DialogueStateTracker.from_events(
        sender_id=uuid.uuid4().hex,
        evts=[
            ActionExecuted(
                "action_session_start",
                timestamp=time.time(),
                metadata={METADATA_SESSION_ID: "s1"},
            ),
            UserUttered(
                "hi",
                timestamp=time.time(),
                metadata={METADATA_SESSION_ID: "s1"},
            ),
        ],
        slots=pii_domain.slots,
    )
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=in_memory_tracker_store,
        lock_store=lock_store,
    )
    await privacy_manager.tracker_store.save(active_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    mock_delete = AsyncMock()
    mock_save = AsyncMock()
    monkeypatch.setattr(privacy_manager.tracker_store, "delete", mock_delete)
    monkeypatch.setattr(privacy_manager.tracker_store, "save", mock_save)

    await privacy_manager._run_tracker_store_anonymization()

    mock_delete.assert_not_called()
    mock_save.assert_not_called()
    privacy_manager.stop()


@pytest.mark.asyncio
async def test_run_tracker_store_deletion_processes_multi_session_tracker_env_set(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """When tracker has multiple sessions and full tracker is ineligible (recent last
    event), we still run per-session logic and drop old sessions."""
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    write_endpoint_config_to_yaml(tmp_path, deletion_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    base = time.time()
    # s1 old (past threshold), s2 recent; full tracker last event recent -> ineligible
    evts = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 200,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered(
            "old session",
            timestamp=base - 200,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        ActionExecuted(
            "action_session_start",
            timestamp=base - 10,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        UserUttered(
            "recent session",
            timestamp=base - 5,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
    ]
    multi_tracker = DialogueStateTracker.from_events(
        "sender", evts, slots=pii_domain.slots
    )
    with pytest.warns(
        FutureWarning, match=USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
    ):
        privacy_manager = await BackgroundPrivacyManager.create_instance(
            Configuration.initialise_endpoints(
                endpoints_path=tmp_path / "endpoints.yml"
            ).endpoints,
            in_memory_tracker_store=in_memory_tracker_store,
            lock_store=lock_store,
        )
    await privacy_manager.tracker_store.save(multi_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    assert privacy_manager._is_tracker_eligible_for_privacy_job(
        multi_tracker, "deletion"
    )
    assert privacy_manager._tracker_has_multiple_sessions(multi_tracker)

    await privacy_manager._run_tracker_store_deletion()

    # Should have saved a tracker with only the recent session (s2)
    saved = await privacy_manager.tracker_store.retrieve_full_tracker("sender")
    assert saved is not None
    user_events = [e for e in saved.events if isinstance(e, UserUttered)]
    assert len(user_events) == 1
    assert user_events[0].text == "recent session"
    privacy_manager.stop()


async def test_run_tracker_store_deletion_processes_multi_session_tracker_env_unset(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    lock_store: InMemoryLockStore,
) -> None:
    """When tracker has multiple sessions and full tracker is ineligible
    (is not terminated by SessionEnded), we return the tracker unchanged."""
    monkeypatch.delenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, raising=False)

    write_endpoint_config_to_yaml(tmp_path, deletion_privacy_config_data)
    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    base = time.time()
    # s1 old (past threshold), s2 recent; full tracker last event recent -> ineligible
    evts = [
        ActionExecuted(
            "action_session_start",
            timestamp=base - 200,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        UserUttered(
            "old session",
            timestamp=base - 200,
            metadata={METADATA_SESSION_ID: "s1"},
        ),
        ActionExecuted(
            "action_session_start",
            timestamp=base - 10,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
        UserUttered(
            "recent session",
            timestamp=base - 5,
            metadata={METADATA_SESSION_ID: "s2"},
        ),
    ]
    multi_tracker = DialogueStateTracker.from_events(
        "sender", evts, slots=pii_domain.slots
    )
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=in_memory_tracker_store,
        lock_store=lock_store,
    )
    await privacy_manager.tracker_store.save(multi_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    assert privacy_manager._tracker_has_multiple_sessions(multi_tracker)
    assert not privacy_manager._is_tracker_eligible_for_privacy_job(
        multi_tracker, "deletion"
    )

    await privacy_manager._run_tracker_store_deletion()
    saved = await privacy_manager.tracker_store.retrieve_full_tracker("sender")
    assert saved.events == multi_tracker.events

    privacy_manager.stop()


@pytest.mark.parametrize(
    "json_exception",
    [
        JsonPatchException("invalid patch in stack"),
        JsonPointerException("invalid path in stack update"),
    ],
)
async def test_run_tracker_store_deletion_handles_json_exceptions_when_events_are_retained(  # noqa: E501
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
    lock_store: LockStore,
    json_exception: Exception,
) -> None:
    """When from_events raises JsonPatchException during deletion, tracker is left unchanged and error is logged."""  # noqa: E501
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")
    write_endpoint_config_to_yaml(
        tmp_path,
        deletion_privacy_config_data,
    )
    # Tracker with recent events so events_to_be_retained is non-empty
    recent_events = [
        ActionExecuted(
            "action_session_start", timestamp=datetime.datetime.now().timestamp()
        ),
        ActionExecuted("action_listen", timestamp=datetime.datetime.now().timestamp()),
        UserUttered(
            "I want to check my account balance.",
            timestamp=datetime.datetime.now().timestamp(),
        ),
    ]
    unanonymized_tracker.update_with_events(recent_events)
    original_events_count = len(list(unanonymized_tracker.events))

    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        Configuration.initialise_endpoints(
            endpoints_path=tmp_path / "endpoints.yml"
        ).endpoints,
        in_memory_tracker_store=in_memory_tracker_store,
        lock_store=lock_store,
    )
    await privacy_manager.tracker_store.save(unanonymized_tracker)
    privacy_manager.tracker_store.domain = pii_domain

    def from_events_raising_json_exception(
        *args: Any, **kwargs: Any
    ) -> DialogueStateTracker:
        raise json_exception

    monkeypatch.setattr(
        "rasa.privacy.privacy_manager.DialogueStateTracker",
        MagicMock(from_events=from_events_raising_json_exception),
    )

    with capture_logs() as caplog:
        await privacy_manager._run_tracker_store_deletion()
        logs = filter_logs(
            caplog,
            "rasa.privacy_manager.error_reconstructing_tracker_after_deletion",
            "error",
        )
        assert len(logs) == 1
        assert "Could not reconstruct tracker" in logs[0].get("event_info", "")

    # No data loss: tracker still in store and unchanged
    tracker = await privacy_manager.tracker_store.retrieve_full_tracker(
        unanonymized_tracker.sender_id
    )
    assert tracker is not None
    assert len(list(tracker.events)) == original_events_count

    privacy_manager.stop()
