import datetime
import time
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, call

import freezegun
import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.available_endpoints import AvailableEndpoints
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.tracker_stores.tracker_store import (
    InMemoryTrackerStore,
)
from rasa.privacy.constants import USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME
from rasa.privacy.privacy_config import PrivacyConfig
from rasa.privacy.privacy_filter import PrivacyFilter
from rasa.privacy.privacy_manager import BackgroundPrivacyManager
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, BotUttered, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import RasaException
from tests.conftest import write_endpoint_config_to_yaml
from tests.utilities import filter_logs

freezegun.config.configure(extend_ignore_list=["transformers"])


@pytest.fixture
def available_endpoints_for_pii_management() -> AvailableEndpoints:
    return AvailableEndpoints.read_endpoints(
        "data/test_privacy/endpoints_with_valid_privacy.yml"
    )


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
    )
    yield privacy_manager
    privacy_manager.stop()


@pytest.fixture
async def empty_privacy_manager(
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


async def test_privacy_manager_run_tracker_store_deletion(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
) -> None:
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")
    write_endpoint_config_to_yaml(
        tmp_path,
        deletion_privacy_config_data,
    )

    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints.read_endpoints(str(tmp_path / "endpoints.yml")),
        in_memory_tracker_store=in_memory_tracker_store,
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


async def test_privacy_manager_run_tracker_store_deletion_retained_events(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    deletion_privacy_config_data: Dict[str, Any],
) -> None:
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
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

    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints.read_endpoints(str(tmp_path / "endpoints.yml")),
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


async def test_privacy_manager_run_tracker_store_anonymization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    pii_domain: Domain,
    anonymization_privacy_config_data: Dict[str, Any],
) -> None:
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")
    write_endpoint_config_to_yaml(
        tmp_path,
        anonymization_privacy_config_data,
    )

    in_memory_tracker_store = InMemoryTrackerStore(Domain.empty())
    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints.read_endpoints(str(tmp_path / "endpoints.yml")),
        in_memory_tracker_store=in_memory_tracker_store,
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


async def test_privacy_manager_run_tracker_store_anonymization_already_anonymized(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    pii_domain: Domain,
    anonymization_privacy_config_data,
) -> None:
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
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

    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints.read_endpoints(str(tmp_path / "endpoints.yml")),
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
async def test_privacy_manager_run_tracker_store_background_jobs_sequentially_both(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    mock_run_tracker_store_anonymization: AsyncMock,
    mock_run_tracker_store_deletion: AsyncMock,
    pii_domain: Domain,
    privacy_config_data: Dict[str, Any],
) -> None:
    """
    We freeze time to ensure that the cron jobs for both anonymization and deletion
    are triggered at the same time, allowing us to test the sequential execution
    of both jobs in the `_run_tracker_store_background_jobs_sequentially` method.
    """
    # Given
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, privacy_config_data, "endpoints_sequential.yml"
    )

    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints.read_endpoints(str(endpoints_path)),
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
async def test_privacy_manager_run_tracker_store_jobs_sequentially_anonymization_only(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    unanonymized_tracker: DialogueStateTracker,
    mock_run_tracker_store_anonymization: AsyncMock,
    mock_run_tracker_store_deletion: AsyncMock,
    privacy_config_data: Dict[str, Any],
) -> None:
    """We freeze time to ensure that the anonymization cron job is triggered.

    The deletion cron job won't run because the current time is 12:30.
    """
    # Given
    monkeypatch.setattr(
        "gliner.GLiNER.from_pretrained", MagicMock(side_effect=ImportError)
    )
    monkeypatch.setenv(USER_CHAT_INACTIVITY_IN_MINUTES_ENV_VAR_NAME, "0")

    endpoints_path = write_endpoint_config_to_yaml(
        tmp_path, privacy_config_data, "endpoints_anonymization_only.yml"
    )

    privacy_manager = await BackgroundPrivacyManager.create_instance(
        AvailableEndpoints.read_endpoints(str(endpoints_path)),
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
