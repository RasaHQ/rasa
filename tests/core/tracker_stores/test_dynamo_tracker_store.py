import os
import uuid
from typing import Any

import boto3
import pytest
from moto import mock_aws
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.constants import ENV_SANIC_WORKERS
from rasa.core.tracker_stores.dynamo_tracker_store import DynamoTrackerStore
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Restarted,
    SessionStarted,
    SlotSet,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import EndpointConfig
from tests.core.tracker_stores.conftest import (
    _saved_tracker_with_multiple_session_starts,
    get_or_create_tracker_store,
)
from tests.utilities import filter_logs


@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def mock_dynamodb(aws_credentials):
    """Return a mocked S3 client"""
    with mock_aws():
        yield boto3.client("dynamodb", region_name="us-east-1")


@pytest.mark.asyncio
async def test_dynamo_get_or_create(test_domain: Domain, mock_dynamodb: Any) -> None:
    await get_or_create_tracker_store(DynamoTrackerStore(test_domain))


@pytest.mark.asyncio
async def test_dynamo_tracker_floats(test_domain: Domain, mock_dynamodb: Any) -> None:
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    tracker = await tracker_store.get_or_create_tracker(
        conversation_id, append_action_listen=False
    )

    # save `slot` event with known `float`-type timestamp
    timestamp = 13423.23434623
    tracker.update(SlotSet("key", "val", timestamp=timestamp))
    await tracker_store.save(tracker)

    # retrieve tracker and the event timestamp is retrieved as a `float`
    tracker = await tracker_store.get_or_create_tracker(conversation_id)
    retrieved_timestamp = tracker.events[0].timestamp
    assert isinstance(retrieved_timestamp, float)
    assert retrieved_timestamp == timestamp


def test_dynamo_tracker_create_table_multiple_sanic_workers_error(
    test_domain: Domain,
    monkeypatch: MonkeyPatch,
    mock_dynamodb: Any,
) -> None:
    monkeypatch.setenv(ENV_SANIC_WORKERS, "2")

    with capture_logs() as caplog:
        with pytest.raises(RasaException) as raised_exception:
            DynamoTrackerStore(test_domain)
            assert (
                "DynamoDB table creation is not supported in "
                "case of multiple sanic workers." in str(raised_exception.value)
            )

        logs = filter_logs(
            event="dynamo_tracker_store.table_creation_not_supported_in_multi_worker_mode",
            log_level="error",
            caplog=caplog,
        )
        assert len(logs) == 1


def test_dynamo_tracker_store_connection_error(domain: Domain):
    store = EndpointConfig.from_dict({"type": "dynamo"})

    with pytest.raises(ConnectionException):
        TrackerStore.create(store, domain)


@pytest.mark.asyncio
async def test_dynamo_tracker_store_delete(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    # Given
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)
    await tracker_store.get_or_create_tracker(
        conversation_id,
    )

    # When
    with capture_logs() as caplog:
        await tracker_store.delete(conversation_id)

        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.delete.deleted_tracker",
            log_level="info",
        )
        assert len(logs) == 1
        assert logs[0].get("sender_id") == conversation_id

    retrieved_tracker = await tracker_store.retrieve(conversation_id)
    assert retrieved_tracker is None


async def test_dynamo_tracker_store_delete_no_tracker(mock_dynamodb: Any) -> None:
    with capture_logs() as caplog:
        tracker_store = DynamoTrackerStore(Domain.empty())
        conversation_id = uuid.uuid4().hex
        await tracker_store.delete(conversation_id)
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.delete.no_tracker_for_sender_id",
            log_level="info",
            log_message_parts=[
                f"Could not find tracker for conversation ID '{conversation_id}'."
            ],
        )

        assert len(logs) == 1


async def test_dynamo_tracker_store_update_tracker(mock_dynamodb: Any) -> None:
    # Given
    sender_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(Domain.empty())
    tracker = await _saved_tracker_with_multiple_session_starts(
        tracker_store, sender_id
    )
    new_events = list(tracker.events)[3:] + [
        UserUttered("What's the weather like today?")
    ]
    new_tracker = DialogueStateTracker.from_events(
        sender_id,
        new_events,
    )

    # When
    await tracker_store.update(new_tracker)

    # Then
    updated_tracker = await tracker_store.retrieve(sender_id)
    assert updated_tracker == new_tracker


async def test_dynamo_tracker_store_save_single_user_uttered(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    # Given
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            SessionStarted(),
            ActionExecuted("action_session_start"),
            SlotSet("session_started_metadata", {}),
            ActionExecuted("action_listen"),
            UserUttered("What's the weather like today?"),
        ],
        slots=test_domain.slots,
        domain=test_domain,
    )

    # When
    await tracker_store.save(tracker)

    retrieved_tracker = await tracker_store.retrieve(conversation_id)
    assert retrieved_tracker.current_state(
        EventVerbosity.APPLIED
    ) == tracker.current_state(EventVerbosity.APPLIED)


async def test_dynamo_tracker_store_save_multiple_turns(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    # Given
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            SessionStarted(),
            ActionExecuted("action_session_start"),
            SlotSet("session_started_metadata", {}),
            ActionExecuted("action_listen"),
            UserUttered("What's the weather like today?"),
            BotUttered("The weather is rainy."),
        ],
        slots=test_domain.slots,
        domain=test_domain,
    )

    await tracker_store.save(tracker)

    tracker.update_with_events(
        [
            UserUttered("Can you tell me the time?"),
            BotUttered("Sure, it's 3 PM."),
            ActionExecuted("action_listen"),
        ],
        domain=test_domain,
    )

    # When

    await tracker_store.save(tracker)

    retrieved_tracker = await tracker_store.retrieve(conversation_id)
    assert retrieved_tracker.current_state(
        EventVerbosity.APPLIED
    ) == tracker.current_state(EventVerbosity.APPLIED)


async def test_dynamo_tracker_store_save_multiple_sessions(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    # Given
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            SessionStarted(),
            ActionExecuted("action_session_start"),
            SlotSet("session_started_metadata", {}),
            ActionExecuted("action_listen"),
            UserUttered("What's the weather like today?"),
            BotUttered("The weather is rainy."),
        ],
        slots=test_domain.slots,
        domain=test_domain,
    )

    await tracker_store.save(tracker)

    tracker.update_with_events(
        [
            UserUttered("/restart"),
            ActionExecuted("action_restart"),
            Restarted(),
            SessionStarted(),
            ActionExecuted("action_session_start"),
            SlotSet("session_started_metadata", {}),
            ActionExecuted("action_listen"),
        ],
        domain=test_domain,
    )

    # When
    await tracker_store.save(tracker)

    retrieved_tracker = await tracker_store.retrieve_full_tracker(conversation_id)
    assert retrieved_tracker.current_state(EventVerbosity.ALL) == tracker.current_state(
        EventVerbosity.ALL
    )
