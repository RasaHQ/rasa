import os
import time
import uuid
from typing import Any, Optional, Text
from unittest.mock import patch

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.constants import ENV_SANIC_WORKERS
from rasa.core.tracker_stores.dynamo_tracker_store import (
    DynamoTrackerStore,
    _deduplicate_events,
)
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.shared.constants import DEFAULT_SENDER_ID, DEFAULT_USER_ID
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    ConversationInactive,
    DialogueStackUpdated,
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
    create_multiple_trackers_with_user_id,
    create_tracker_with_user_id,
    create_trackers_with_same_timestamp,
    get_or_create_tracker_store,
    old_tracker_gets_timestamp_on_save,
    old_tracker_gets_timestamp_on_update,
    sort_key,
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
    """Return a mocked S3 client."""
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


@pytest.mark.asyncio
async def test_dynamo_save_put_item_non_conditional_client_error_is_reraised(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Non-ConditionalCheckFailedException from put_item on new item is re-raised.

    The save path calls ``put_item`` with ``attribute_not_exists(sender_id)`` for
    new items.  A ``ConditionalCheckFailedException`` is swallowed (item already
    exists — fall through to update).  Any other ``ClientError`` (e.g.
    ``ProvisionedThroughputExceededException``) must propagate to the caller.
    """
    tracker_store = DynamoTrackerStore(test_domain)
    tracker = DialogueStateTracker.from_events(
        "test_sender",
        [SessionStarted(), UserUttered("hello")],
        slots=test_domain.slots,
    )

    throughput_error = ClientError(
        error_response={
            "Error": {
                "Code": "ProvisionedThroughputExceededException",
                "Message": "Rate exceeded",
            }
        },
        operation_name="PutItem",
    )

    with patch.object(tracker_store.db, "put_item", side_effect=throughput_error):
        with pytest.raises(ClientError) as exc_info:
            await tracker_store.save(tracker)

    assert (
        exc_info.value.response["Error"]["Code"]
        == "ProvisionedThroughputExceededException"
    )


@pytest.mark.asyncio
async def test_dynamo_save_update_item_non_conditional_client_error_is_reraised(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Non-ConditionalCheckFailedException from update_item on existing item is re-raised.

    After the first ``put_item`` succeeds (item exists), subsequent saves call
    ``update_item`` to append new events.  A ``ConditionalCheckFailedException``
    is swallowed (duplicate event — already stored).  Any other ``ClientError``
    must propagate to the caller.
    """
    tracker_store = DynamoTrackerStore(test_domain)
    tracker = DialogueStateTracker.from_events(
        "test_sender",
        [SessionStarted(), UserUttered("hello")],
        slots=test_domain.slots,
    )
    # First save succeeds — item now exists in the table.
    await tracker_store.save(tracker)

    tracker.update(UserUttered("world"))

    throughput_error = ClientError(
        error_response={
            "Error": {
                "Code": "ProvisionedThroughputExceededException",
                "Message": "Rate exceeded",
            }
        },
        operation_name="UpdateItem",
    )

    with patch.object(tracker_store.db, "update_item", side_effect=throughput_error):
        with pytest.raises(ClientError) as exc_info:
            await tracker_store.save(tracker)

    assert (
        exc_info.value.response["Error"]["Code"]
        == "ProvisionedThroughputExceededException"
    )


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


async def test_dynamo_tracker_store_retrieve_widens_prefix_for_stack_integrity_true_mode(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    tracker_store = DynamoTrackerStore(test_domain)
    sender_id = "dynamo_widen_true"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
            ),
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "SECOND"}]'
            ),
        ],
    )
    await tracker_store.save(tracker)

    retrieved = await tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 4
    assert retrieved.stack.frames[0].step_id == "SECOND"


async def test_dynamo_tracker_store_retrieve_widens_prefix_for_stack_integrity_false_mode(
    mock_dynamodb: Any,
) -> None:
    domain = Domain.from_dict({"session_config": {"start_session_after_expiry": False}})
    tracker_store = DynamoTrackerStore(domain)
    sender_id = "dynamo_widen_false"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
            ),
            ConversationInactive(),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "AFTER_INACTIVE"}]'
            ),
        ],
    )
    await tracker_store.save(tracker)

    retrieved = await tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 3
    assert retrieved.stack.frames[0].step_id == "AFTER_INACTIVE"


async def test_dynamo_tracker_store_save_single_user_uttered(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    # Given
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted("action_session_start"),
            SessionStarted(),
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
            ActionExecuted("action_session_start"),
            SessionStarted(),
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


@pytest.mark.asyncio
async def test_dynamo_tracker_store_save_is_idempotent(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Calling save() twice for the same turn does not duplicate events.

    This covers the race-condition / retry scenario where a duplicate save call
    would previously append the same last-turn events a second time, causing
    JsonPatchConflict errors during replay.
    """
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted("action_session_start"),
            SessionStarted(),
            SlotSet("session_started_metadata", {}),
            ActionExecuted("action_listen"),
            UserUttered("hello"),
            BotUttered("hi there"),
            ActionExecuted("action_listen"),
        ],
        slots=test_domain.slots,
        domain=test_domain,
    )

    # Simulate a duplicate save (retry / race condition).
    await tracker_store.save(tracker)
    events_after_first_save = len(
        tracker_store.db.get_item(Key={"sender_id": conversation_id})
        .get("Item", {})
        .get("events", [])
    )

    with capture_logs() as caplog:
        await tracker_store.save(tracker)
        debug_log = filter_logs(
            caplog,
            "rasa.core.tracker_stores.dynamo_tracker_store.save.item_already_exists",
            "debug",
        )
        assert len(debug_log) == 1
        warning_log = filter_logs(
            caplog,
            "rasa.core.tracker_stores.dynamo_tracker_store.save.duplicate_events_skipped",
        )
        assert len(warning_log) == 1

    # The raw event count in DynamoDB must not grow on the duplicate save.
    item = tracker_store.db.get_item(Key={"sender_id": conversation_id}).get("Item", {})
    events_after_second_save = len(item.get("events", []))
    assert (
        events_after_second_save == events_after_first_save
    ), "Duplicate save() appended events a second time — idempotency broken."

    # The retrieved tracker state must be consistent (not corrupted by a double-append).
    retrieved = await tracker_store.retrieve(conversation_id)
    assert retrieved is not None
    assert retrieved.latest_message.text == "hello"


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


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_dynamo_tracker_store_preserves_user_id(
    test_domain: Domain, mock_dynamodb: Any, user_id: Optional[Text]
) -> None:
    """Test DynamoTrackerStore preserves user_id on save/retrieve."""
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
            UserUttered("Hello"),
        ],
        slots=test_domain.slots,
        domain=test_domain,
        user_id=user_id,
    )

    # When
    await tracker_store.save(tracker)

    # Then
    retrieved_tracker = await tracker_store.retrieve(conversation_id)
    assert retrieved_tracker is not None
    assert retrieved_tracker.user_id == user_id

    # Test with retrieve_full_tracker as well
    retrieved_full_tracker = await tracker_store.retrieve_full_tracker(conversation_id)
    assert retrieved_full_tracker is not None
    assert retrieved_full_tracker.user_id == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_dynamo_tracker_store_get_or_create(
    test_domain: Domain, mock_dynamodb: Any, user_id: Optional[Text]
) -> None:
    """Test get_or_create_tracker with and without user_id."""
    # Given
    tracker_store = DynamoTrackerStore(test_domain)

    # When
    tracker = await tracker_store.get_or_create_tracker(
        DEFAULT_SENDER_ID, user_id=user_id
    )

    # Then
    assert tracker.user_id == user_id

    retrieved = await tracker_store.get_or_create_tracker(DEFAULT_SENDER_ID)
    assert retrieved.user_id == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_dynamo_tracker_store_update_preserves_user_id(
    test_domain: Domain, mock_dynamodb: Any, user_id: Optional[Text]
) -> None:
    """Test update operation preserves user_id."""
    # Given
    tracker_store = DynamoTrackerStore(test_domain)

    # Create and save initial tracker
    tracker = DialogueStateTracker.from_events(
        DEFAULT_SENDER_ID,
        [SessionStarted(), UserUttered("Hello")],
        slots=test_domain.slots,
        domain=test_domain,
        user_id=user_id,
    )
    await tracker_store.save(tracker)

    # When
    await tracker_store.update(tracker)

    # Then
    retrieved = await tracker_store.retrieve(DEFAULT_SENDER_ID)

    assert retrieved is not None
    assert retrieved.user_id == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_dynamo_tracker_store_save_appends_events_preserves_user_id(
    test_domain: Domain, mock_dynamodb: Any, user_id: Optional[Text]
) -> None:
    """Test incremental saves preserve user_id."""
    # Given
    tracker_store = DynamoTrackerStore(test_domain)

    # Create initial tracker
    tracker = DialogueStateTracker.from_events(
        DEFAULT_SENDER_ID,
        [SessionStarted(), UserUttered("Hello")],
        slots=test_domain.slots,
        domain=test_domain,
        user_id=user_id,
    )
    await tracker_store.save(tracker)

    # When
    tracker.update_with_events(
        [ActionExecuted("action_listen"), BotUttered("Hi")],
        domain=test_domain,
    )
    await tracker_store.save(tracker)

    # Then
    retrieved = await tracker_store.retrieve(DEFAULT_SENDER_ID)
    assert retrieved is not None
    assert retrieved.user_id == user_id


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_scanning_fallback(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id falls back to scanning when GSI doesn't exist."""
    # Given
    user_id = "user_123"
    conversation_id_1 = uuid.uuid4().hex
    conversation_id_2 = uuid.uuid4().hex
    conversation_id_3 = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)

    # Create trackers with user_id
    await create_tracker_with_user_id(
        tracker_store, conversation_id_1, user_id, domain=test_domain
    )
    await create_tracker_with_user_id(
        tracker_store,
        conversation_id_2,
        user_id,
        [SessionStarted(), UserUttered("Hi")],
        domain=test_domain,
    )

    # Create tracker with different user_id
    await create_tracker_with_user_id(
        tracker_store,
        conversation_id_3,
        "user_456",
        [SessionStarted(), UserUttered("Hey")],
        domain=test_domain,
    )

    # When - GSI doesn't exist, should fall back to scanning
    with capture_logs() as caplog:
        retrieved_trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(retrieved_trackers) == 2
        sender_ids = {t["sender_id"] for t in retrieved_trackers}
        assert conversation_id_1 in sender_ids
        assert conversation_id_2 in sender_ids
        assert conversation_id_3 not in sender_ids

        # Verify all trackers have correct user_id
        assert all(t["user_id"] == user_id for t in retrieved_trackers)

        # Verify fallback log message
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 1


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_no_matching_trackers_scanning_fallback(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id returns empty list when no trackers match."""
    # Given
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)

    # Create tracker with different user_id
    await create_tracker_with_user_id(
        tracker_store, conversation_id, "user_456", domain=test_domain
    )

    # When
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 0
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 1


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_filters_trackers_without_user_id_scanning(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id filters out trackers without user_id.

    Uses scanning fallback when GSI doesn't exist.
    """
    # Given
    user_id = "user_123"
    conversation_id_1 = uuid.uuid4().hex
    conversation_id_2 = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)

    # Create tracker with user_id
    await create_tracker_with_user_id(
        tracker_store, conversation_id_1, user_id, domain=test_domain
    )

    # Create tracker without user_id
    await create_tracker_with_user_id(
        tracker_store,
        conversation_id_2,
        None,
        [SessionStarted(), UserUttered("Hi")],
        domain=test_domain,
    )

    # When
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert trackers[0]["sender_id"] == conversation_id_1
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 1


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_save_sets_user_id_scanning_fallback(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that save method sets user_id in DynamoDB during update_item operations."""
    # Given
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)

    # Create initial tracker without user_id (simulating pre-migration item)
    tracker = await create_tracker_with_user_id(
        tracker_store, conversation_id, None, domain=test_domain
    )
    # Ensure tracker doesn't have user_id attribute
    assert tracker.user_id is None

    # Verify item was created without user_id
    retrieved_before = await tracker_store.retrieve_full_tracker(conversation_id)
    assert retrieved_before is not None
    assert retrieved_before.user_id is None

    # Add user_id and update tracker with new events (triggers update_item path)
    tracker.user_id = user_id
    tracker.update_with_events(
        [UserUttered("How are you?")],
        domain=test_domain,
    )
    await tracker_store.save(tracker)

    # When
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert trackers[0]["sender_id"] == conversation_id
        assert trackers[0]["user_id"] == user_id

        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 1


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_update_sets_user_id_scanning_fallback(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that update method sets user_id in DynamoDB via put_item."""
    # Given
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)

    # Create tracker with user_id
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [SessionStarted(), UserUttered("Hello")],
        slots=test_domain.slots,
        domain=test_domain,
    )
    await tracker_store.save(tracker)

    tracker.user_id = user_id
    tracker.update_with_events(
        [UserUttered("How are you?")],
        domain=test_domain,
    )
    await tracker_store.update(tracker)

    # When
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert trackers[0]["sender_id"] == conversation_id
        assert trackers[0]["user_id"] == user_id
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 1


def create_user_id_gsi(tracker_store: DynamoTrackerStore) -> None:
    """This simulates the manual GSI creation that would be done in production.

    Creates a GSI with:
    - Partition key: user_id (String)
    - Sort key: conversation_started_timestamp (Number)
    - Index name: user_id-index

    This enables efficient database-level sorting and pagination.
    """
    dynamodb_resource = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb_resource.Table(tracker_store.table_name)

    # Wait for table to exist
    table.meta.client.get_waiter("table_exists").wait(
        TableName=tracker_store.table_name
    )

    # Create GSI with user_id as partition key and
    # conversation_started_timestamp as sort key
    try:
        table.update(
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {
                    "AttributeName": "conversation_started_timestamp",
                    "AttributeType": "N",
                },
            ],
            GlobalSecondaryIndexUpdates=[
                {
                    "Create": {
                        "IndexName": "user_id-index",
                        "KeySchema": [
                            {"AttributeName": "user_id", "KeyType": "HASH"},
                            {
                                "AttributeName": "conversation_started_timestamp",
                                "KeyType": "RANGE",
                            },
                        ],
                        "Projection": {"ProjectionType": "ALL"},
                        "ProvisionedThroughput": {
                            "ReadCapacityUnits": 5,
                            "WriteCapacityUnits": 5,
                        },
                    }
                }
            ],
        )
        # Wait for GSI to be active
        max_wait = 30
        wait_time = 0
        while wait_time < max_wait:
            table.reload()
            gsi_status = None
            for gsi in table.global_secondary_indexes or []:
                if gsi["IndexName"] == "user_id-index":
                    gsi_status = gsi["IndexStatus"]
                    break
            if gsi_status == "ACTIVE":
                break
            time.sleep(0.5)
            wait_time += 0.5
    except Exception:
        # GSI might already exist or creation failed, continue with test
        # (fallback behavior will be tested)
        pass


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id using GSI when available."""
    # Given
    user_id = "user_123"
    conversation_id_1 = uuid.uuid4().hex
    conversation_id_2 = uuid.uuid4().hex
    conversation_id_3 = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)

    # Create GSI on user_id for testing
    create_user_id_gsi(tracker_store)

    # Create trackers with user_id
    await create_tracker_with_user_id(
        tracker_store, conversation_id_1, user_id, domain=test_domain
    )
    await create_tracker_with_user_id(
        tracker_store, conversation_id_2, user_id, domain=test_domain
    )

    # Create tracker with different user_id
    await create_tracker_with_user_id(
        tracker_store, conversation_id_3, "user_456", domain=test_domain
    )

    # When - should use GSI
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 2
        sender_ids = {tracker["sender_id"] for tracker in trackers}
        assert conversation_id_1 in sender_ids
        assert conversation_id_2 in sender_ids
        assert conversation_id_3 not in sender_ids

        # Verify all trackers have correct user_id
        assert all(t["user_id"] == user_id for t in trackers)

        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_no_matching_trackers(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id with GSI returns empty list when no trackers match."""
    # Given
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create tracker with different user_id
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [SessionStarted(), UserUttered("Hello")],
        slots=test_domain.slots,
        domain=test_domain,
        user_id="user_456",
    )
    await tracker_store.save(tracker)

    # When - should use GSI
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 0

        # Verify GSI was used (no fallback log)
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_filters_trackers_without_user_id(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id with GSI filters out trackers without user_id."""
    # Given
    user_id = "user_123"
    conversation_id_1 = uuid.uuid4().hex
    conversation_id_2 = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create tracker with user_id
    await create_tracker_with_user_id(
        tracker_store, conversation_id_1, user_id, domain=test_domain
    )

    # Create tracker without user_id (won't be indexed in GSI)
    await create_tracker_with_user_id(
        tracker_store, conversation_id_2, None, domain=test_domain
    )

    # When - should use GSI (only returns items with user_id in GSI)
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert trackers[0]["sender_id"] == conversation_id_1
        assert trackers[0]["user_id"] == user_id

        # Verify GSI was used
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_save_sets_user_id(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that save method sets user_id in DynamoDB with GSI indexing."""
    # Given
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create initial tracker without user_id
    tracker = await create_tracker_with_user_id(
        tracker_store, conversation_id, None, domain=test_domain
    )
    assert tracker.user_id is None

    # Add user_id and update tracker with new events
    tracker.user_id = user_id
    tracker.update_with_events(
        [UserUttered("How are you?")],
        domain=test_domain,
    )
    await tracker_store.save(tracker)

    # When - should use GSI to find the tracker
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert trackers[0]["sender_id"] == conversation_id
        assert trackers[0]["user_id"] == user_id

        # Verify GSI was used
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_update_sets_user_id(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that update method sets user_id in DynamoDB with GSI indexing."""
    # Given
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create tracker with user_id
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [SessionStarted(), UserUttered("Hello")],
        slots=test_domain.slots,
        domain=test_domain,
    )
    await tracker_store.save(tracker)

    tracker.user_id = user_id
    tracker.update_with_events(
        [UserUttered("How are you?")],
        domain=test_domain,
    )
    await tracker_store.update(tracker)

    # When - should use GSI to find the tracker
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert trackers[0]["sender_id"] == conversation_id
        assert trackers[0]["user_id"] == user_id

        # Verify GSI was used
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_pagination(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id with GSI handles multiple trackers correctly."""
    # Given
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create multiple trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 40, domain=test_domain
    )
    conversation_ids = {tracker.sender_id for tracker in saved_trackers}

    # When - should use GSI and handle all trackers
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 40
        retrieved_ids = {tracker["sender_id"] for tracker in trackers}
        assert retrieved_ids == conversation_ids

        # Verify all trackers have correct user_id
        assert all(t["user_id"] == user_id for t in trackers)

        # Verify GSI was used
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_limit(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id with GSI respects limit parameter."""
    # Given
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create multiple trackers with the same user_id
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=test_domain
    )

    # When - request only 5 trackers
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=5)

        # Then
        assert len(trackers) == 5
        assert all(t["user_id"] == user_id for t in trackers)

        # Verify GSI was used
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_skip(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id with GSI respects skip parameter."""
    # Given
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create multiple trackers with the same user_id
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=test_domain
    )

    # When - skip first 3 trackers
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=3)

        # Then
        assert len(trackers) == 7  # 10 total - 3 skipped
        assert all(t["user_id"] == user_id for t in trackers)

        # Verify GSI was used
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_with_gsi_skip_and_limit(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id with GSI respects both skip and limit parameters."""
    # Given
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    # Create multiple trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=test_domain
    )

    # When - skip first 2, then return next 3
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=2, limit=3)

        # Then
        assert len(trackers) == 3

        saved_trackers_sorted = sorted(saved_trackers, key=sort_key)
        expected_sender_ids = [t.sender_id for t in saved_trackers_sorted[2:5]]
        assert [
            t["sender_id"] for t in trackers
        ] == expected_sender_ids, "Retrieved trackers should match expected order"
        assert all(t["user_id"] == user_id for t in trackers)
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_trackers_by_user_id_scanning_fallback_with_pagination(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test get_trackers_by_user_id pagination works with scanning fallback."""
    # Given
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)

    # Create multiple trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=test_domain
    )

    # When - GSI doesn't exist, should fall back to scanning with pagination
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=2, limit=3)

        # Then
        assert len(trackers) == 3

        saved_trackers_sorted = sorted(saved_trackers, key=sort_key)
        expected_sender_ids = [t.sender_id for t in saved_trackers_sorted[2:5]]
        assert [
            t["sender_id"] for t in trackers
        ] == expected_sender_ids, "Retrieved trackers should match expected order"
        assert all(t["user_id"] == user_id for t in trackers)

        # Verify fallback log message
        logs = filter_logs(
            caplog,
            event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
            log_level="debug",
        )
        assert len(logs) == 1


# Backward compatibility tests for conversation_started_timestamp
@pytest.mark.asyncio
async def test_dynamo_old_tracker_gets_timestamp_on_save(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that old tracker without conversation_started_timestamp gets it on save."""
    tracker_store = DynamoTrackerStore(test_domain)
    await old_tracker_gets_timestamp_on_save(tracker_store, domain=test_domain)


@pytest.mark.asyncio
async def test_dynamo_old_tracker_gets_timestamp_on_update(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that old tracker without conversation_started_timestamp gets it
    on update."""
    tracker_store = DynamoTrackerStore(test_domain)
    await old_tracker_gets_timestamp_on_update(tracker_store, domain=test_domain)


# Sorting consistency tests
@pytest.mark.asyncio
async def test_dynamo_sorting_by_sender_id_when_timestamps_identical(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that trackers with identical timestamps are sorted by sender_id."""
    tracker_store = DynamoTrackerStore(test_domain)
    user_id = "user_123"
    timestamp = 1234567890.0
    sender_ids = ["sender_c", "sender_a", "sender_b"]

    # Create trackers with same timestamp
    await create_trackers_with_same_timestamp(
        tracker_store, user_id, timestamp, sender_ids, domain=test_domain
    )

    # Retrieve and verify sorting
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Should be sorted by sender_id when timestamps are identical
    assert len(trackers) == 3
    assert trackers[0]["sender_id"] == "sender_a"
    assert trackers[1]["sender_id"] == "sender_b"
    assert trackers[2]["sender_id"] == "sender_c"

    # All should have same timestamp
    for tracker in trackers:
        assert tracker["conversation_started_timestamp"] == timestamp


@pytest.mark.asyncio
async def test_dynamo_pagination_very_large_skip(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that very large skip values are handled gracefully."""
    tracker_store = DynamoTrackerStore(test_domain)
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=test_domain
    )

    # Very large skip should return empty list
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=1000000)

    assert len(trackers) == 0


@pytest.mark.asyncio
async def test_dynamo_pagination_very_large_limit(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that very large limit values are handled gracefully."""
    tracker_store = DynamoTrackerStore(test_domain)
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=test_domain
    )

    # Very large limit should return all items (up to available)
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=1000000)

    assert len(trackers) == 5


@pytest.mark.asyncio
async def test_dynamo_negative_skip_and_limit_ignored(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Test that both negative skip and limit values are ignored."""
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=test_domain
    )

    # When: Retrieve with both negative skip and limit
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=-3, limit=-2)

    # Then: Should return all trackers (both negative values ignored)
    assert len(trackers) == 5
    assert all(t["user_id"] == user_id for t in trackers)


# ---------------------------------------------------------------------------
# _deduplicate_events unit tests
# ---------------------------------------------------------------------------


def test_deduplicate_events_no_duplicates() -> None:
    """Events without duplicates are returned unchanged."""
    events = [
        {"event": "user", "timestamp": 1.0, "text": "hello"},
        {"event": "bot", "timestamp": 2.0, "text": "hi"},
        {"event": "slot", "timestamp": 3.0, "name": "x", "value": 1},
    ]
    assert _deduplicate_events(events) == events


def test_deduplicate_events_exact_duplicates_removed() -> None:
    """Exact duplicate events are collapsed to a single occurrence."""
    event = {"event": "user", "timestamp": 1.0, "text": "hello"}
    assert _deduplicate_events([event, event]) == [event]


def test_deduplicate_events_preserves_first_occurrence_order() -> None:
    """First occurrence of each event is kept; later duplicates are dropped."""
    a = {"event": "user", "timestamp": 1.0, "text": "a"}
    b = {"event": "bot", "timestamp": 2.0, "text": "b"}
    result = _deduplicate_events([a, b, a, b])
    assert result == [a, b]


def test_deduplicate_events_same_timestamp_different_slot_names_kept() -> None:
    """Multiple slot events at the same timestamp with different names are all kept."""
    slot1 = {"event": "slot", "timestamp": 1.0, "name": "full_name", "value": None}
    slot2 = {"event": "slot", "timestamp": 1.0, "name": "email", "value": None}
    slot3 = {"event": "slot", "timestamp": 1.0, "name": "phone", "value": None}
    result = _deduplicate_events([slot1, slot2, slot3])
    assert result == [slot1, slot2, slot3]


def test_deduplicate_events_duplicate_block_at_end_removed() -> None:
    """A duplicate block appended at the end (e.g. due to a double save) is stripped."""
    turn1 = [
        {"event": "user", "timestamp": 1.0, "text": "hi"},
        {"event": "bot", "timestamp": 1.1, "text": "hello"},
    ]
    turn2 = [
        {"event": "user", "timestamp": 2.0, "text": "bye"},
        {"event": "bot", "timestamp": 2.1, "text": "goodbye"},
    ]
    # Simulate save() called twice for turn2 — turn2 events appear twice.
    duplicated = turn1 + turn2 + turn2
    result = _deduplicate_events(duplicated)
    assert result == turn1 + turn2


def test_deduplicate_events_empty_list() -> None:
    """Empty input returns empty output."""
    assert _deduplicate_events([]) == []


@pytest.mark.asyncio
async def test_retrieve_deduplicates_events_in_dynamodb(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Retrieving a tracker whose DynamoDB record has duplicate events succeeds.

    This reproduces the bug where load-test seeding called save() twice for the
    same turn, causing a JsonPatchConflict when replaying stack-update events.
    """
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)

    # Build a tracker with a stack operation that removes a field.
    # The patch below first adds a regular flow frame (which has frame_type),
    # then converts it to a pattern_completed frame by removing frame_type and
    # adding previous_flow_name.  This mirrors the exact patch shape that
    # triggers the bug in production.
    add_patch = '[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
    remove_frame_type_patch = (
        '[{"op": "remove", "path": "/0/frame_type"},'
        ' {"op": "add", "path": "/0/previous_flow_name", "value": "foo"},'
        ' {"op": "replace", "path": "/0/flow_id", "value": "pattern_completed"},'
        ' {"op": "replace", "path": "/0/frame_id", "value": "NRQPGMKX"},'
        ' {"op": "replace", "path": "/0/step_id", "value": "START"},'
        ' {"op": "replace", "path": "/0/type", "value": "pattern_completed"}]'
    )

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted("action_session_start"),
            SessionStarted(),
            ActionExecuted("action_listen"),
            UserUttered("start"),
            DialogueStackUpdated(update=add_patch),
            UserUttered("done"),
            DialogueStackUpdated(update=remove_frame_type_patch),
        ],
        slots=test_domain.slots,
        domain=test_domain,
    )

    # Store the full tracker directly (bypassing the save() partial-turn
    # optimisation) so that all events land in DynamoDB.
    serialized = DynamoTrackerStore.serialise_tracker(tracker)
    tracker_store.db.put_item(Item=serialized)

    # Manually inject duplicate events into DynamoDB to reproduce the bug.
    # save() called twice for the same turn appends the last-turn events twice.
    item = tracker_store.db.get_item(Key={"sender_id": conversation_id}).get("Item", {})
    original_events = item.get("events", [])
    # Find the index of the second (last) UserUttered event and duplicate from there.
    last_user_idx = max(
        i for i, e in enumerate(original_events) if e.get("event") == "user"
    )
    duplicated_events = original_events + original_events[last_user_idx:]
    tracker_store.db.update_item(
        Key={"sender_id": conversation_id},
        UpdateExpression="SET events = :events",
        ExpressionAttributeValues={":events": duplicated_events},
    )

    # When: retrieve must not raise JsonPatchConflict
    retrieved = await tracker_store.retrieve_full_tracker(conversation_id)

    # Then: tracker is valid and matches the original
    assert retrieved is not None
    assert retrieved.sender_id == conversation_id


# ---------------------------------------------------------------------------
# get_serialized_trackers_by_user_id tests
# ---------------------------------------------------------------------------


def _assert_serialized_tracker_format(
    tracker_dict: dict, expected_sender_id: str, expected_user_id: str
) -> None:
    """Assert a serialized tracker dict has the expected events-centric format."""
    assert tracker_dict["sender_id"] == expected_sender_id
    assert tracker_dict["user_id"] == expected_user_id
    assert isinstance(tracker_dict["events"], list)
    assert len(tracker_dict["events"]) > 0
    assert "conversation_started_timestamp" in tracker_dict
    assert "current_session_id" in tracker_dict


@pytest.mark.asyncio
async def test_get_serialized_trackers_by_user_id_with_gsi(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """GSI path returns correct events-centric dicts filtered by user_id."""
    user_id = "user_123"
    conversation_id_1 = uuid.uuid4().hex
    conversation_id_2 = uuid.uuid4().hex
    conversation_id_3 = uuid.uuid4().hex

    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    await create_tracker_with_user_id(
        tracker_store, conversation_id_1, user_id, domain=test_domain
    )
    await create_tracker_with_user_id(
        tracker_store, conversation_id_2, user_id, domain=test_domain
    )
    await create_tracker_with_user_id(
        tracker_store, conversation_id_3, "user_456", domain=test_domain
    )

    with capture_logs() as caplog:
        result = await tracker_store.get_serialized_trackers_by_user_id(user_id)

    assert len(result) == 2
    sender_ids = {t["sender_id"] for t in result}
    assert conversation_id_1 in sender_ids
    assert conversation_id_2 in sender_ids
    assert conversation_id_3 not in sender_ids
    for t in result:
        _assert_serialized_tracker_format(t, t["sender_id"], user_id)
    logs = filter_logs(
        caplog,
        event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
        log_level="debug",
    )
    assert len(logs) == 0


@pytest.mark.asyncio
async def test_get_serialized_trackers_by_user_id_with_gsi_no_matching_trackers(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """GSI path returns empty list when no trackers match the user_id."""
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    tracker = DialogueStateTracker.from_events(
        uuid.uuid4().hex,
        [SessionStarted(), UserUttered("Hello")],
        slots=test_domain.slots,
        domain=test_domain,
        user_id="user_456",
    )
    await tracker_store.save(tracker)

    result = await tracker_store.get_serialized_trackers_by_user_id("user_123")
    assert result == []


@pytest.mark.asyncio
async def test_get_serialized_trackers_by_user_id_with_gsi_skip_and_limit(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """GSI path respects skip and limit pagination parameters."""
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=test_domain
    )

    result = await tracker_store.get_serialized_trackers_by_user_id(
        user_id, skip=2, limit=3
    )

    assert len(result) == 3
    for t in result:
        assert t["user_id"] == user_id
        assert isinstance(t["events"], list)


@pytest.mark.asyncio
async def test_get_serialized_trackers_by_user_id_scanning_fallback(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Fallback scan path returns correct dicts when GSI is unavailable."""
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)  # no GSI created

    await create_tracker_with_user_id(
        tracker_store, conversation_id, user_id, domain=test_domain
    )

    with capture_logs() as caplog:
        result = await tracker_store.get_serialized_trackers_by_user_id(user_id)

    assert len(result) == 1
    _assert_serialized_tracker_format(result[0], conversation_id, user_id)
    logs = filter_logs(
        caplog,
        event="dynamo_tracker_store.get_serialized_trackers_by_user_id.gsi_not_found",
        log_level="debug",
    )
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_get_serialized_trackers_by_user_id_scanning_fallback_with_pagination(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Fallback scan path respects skip and limit pagination."""
    user_id = "user_123"
    tracker_store = DynamoTrackerStore(test_domain)

    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=test_domain
    )

    result = await tracker_store.get_serialized_trackers_by_user_id(
        user_id, skip=2, limit=3
    )

    assert len(result) == 3
    for t in result:
        assert t["user_id"] == user_id
        assert isinstance(t["events"], list)


@pytest.mark.asyncio
async def test_get_serialized_trackers_by_user_id_events_are_plain_dicts(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """Events in the response are plain dicts, not Event objects, and have no Decimals."""
    user_id = "user_123"
    conversation_id = uuid.uuid4().hex
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    await create_tracker_with_user_id(
        tracker_store, conversation_id, user_id, domain=test_domain
    )

    result = await tracker_store.get_serialized_trackers_by_user_id(user_id)

    assert len(result) == 1
    for event in result[0]["events"]:
        assert isinstance(event, dict)
        # Timestamps must be float, not Decimal (Decimal would break JSON serialization)
        if "timestamp" in event:
            assert isinstance(event["timestamp"], float)


@pytest.mark.asyncio
async def test_dynamo_get_or_create_table_reuses_existing_table(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """When the DynamoDB table already exists, get_or_create_table returns it without
    creating a new one (exercises the else branch of the try/except)."""
    store1 = DynamoTrackerStore(test_domain)
    # Second store creation finds the table that store1 already created
    store2 = DynamoTrackerStore(test_domain)
    assert store2.db is not None
    assert store2.table_name == store1.table_name


@pytest.mark.asyncio
async def test_dynamo_keys_pagination_follows_last_evaluated_key(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """keys() issues additional scans when DynamoDB returns a LastEvaluatedKey."""
    tracker_store = DynamoTrackerStore(test_domain)

    first_page = {
        "Items": [{"sender_id": "sender_page_1"}],
        "LastEvaluatedKey": {"sender_id": "sender_page_1"},
    }
    second_page = {"Items": [{"sender_id": "sender_page_2"}]}

    with patch.object(tracker_store.db, "scan", side_effect=[first_page, second_page]):
        keys = list(await tracker_store.keys())

    assert set(keys) == {"sender_page_1", "sender_page_2"}


def test_items_to_serialized_skips_items_without_sender_id(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """_items_to_serialized silently skips DynamoDB items that have no sender_id."""
    tracker_store = DynamoTrackerStore(test_domain)
    items = [
        {"user_id": "user_no_sender", "events": []},  # missing sender_id
        {
            "sender_id": "valid_sender",
            "user_id": "user_no_sender",
            "events": [],
            "conversation_started_timestamp": 1_700_000_000.0,
        },
    ]
    result = tracker_store._items_to_serialized(items)

    assert len(result) == 1
    assert result[0]["sender_id"] == "valid_sender"


@pytest.mark.asyncio
async def test_dynamo_get_serialized_trackers_gsi_pagination(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """get_serialized_trackers_by_user_id() follows GSI LastEvaluatedKey across pages."""
    user_id = "user_gsi_paginate"
    sender_1 = "sender_gsi_p1"
    sender_2 = "sender_gsi_p2"

    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    def _make_item(sid: str, ts: float) -> dict:
        return {
            "sender_id": sid,
            "user_id": user_id,
            "conversation_started_timestamp": ts,
            "events": [],
        }

    first_page = {
        "Items": [_make_item(sender_1, 1_700_000_001.0)],
        "LastEvaluatedKey": {"sender_id": sender_1, "user_id": user_id},
    }
    second_page = {"Items": [_make_item(sender_2, 1_700_000_002.0)]}

    with patch.object(tracker_store.db, "query", side_effect=[first_page, second_page]):
        result = await tracker_store.get_serialized_trackers_by_user_id(user_id)

    assert len(result) == 2
    assert {t["sender_id"] for t in result} == {sender_1, sender_2}


@pytest.mark.asyncio
async def test_dynamo_generic_gsi_exception_triggers_scan_fallback(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """A generic Exception from db.query (not ResourceNotFoundException) is caught,
    logged, and control falls through to the scan-based fallback."""
    user_id = "user_exc_fallback"
    tracker_store = DynamoTrackerStore(test_domain)
    create_user_id_gsi(tracker_store)

    with patch.object(
        tracker_store.db, "query", side_effect=RuntimeError("transient error")
    ):
        with capture_logs() as caplog:
            result = await tracker_store.get_serialized_trackers_by_user_id(user_id)

    # Scan found nothing — the important assertion is that it didn't raise
    assert result == []
    logs = filter_logs(
        caplog,
        event=(
            "dynamo_tracker_store."
            "get_serialized_trackers_by_user_id.gsi_query_failed"
        ),
        log_level="debug",
    )
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_dynamo_fallback_scan_pagination_follows_last_evaluated_key(
    test_domain: Domain, mock_dynamodb: Any
) -> None:
    """_fallback_get_serialized_trackers_by_user_id follows LastEvaluatedKey across
    scan pages."""
    user_id = "user_scan_paginate"
    tracker_store = DynamoTrackerStore(test_domain)

    def _make_item(sid: str, ts: float) -> dict:
        return {
            "sender_id": sid,
            "user_id": user_id,
            "events": [],
            "conversation_started_timestamp": ts,
        }

    first_page = {
        "Items": [_make_item("scan_s1", 1_700_000_001.0)],
        "LastEvaluatedKey": {"sender_id": "scan_s1"},
    }
    second_page = {"Items": [_make_item("scan_s2", 1_700_000_002.0)]}

    with patch.object(tracker_store.db, "scan", side_effect=[first_page, second_page]):
        result = await tracker_store._fallback_get_serialized_trackers_by_user_id(
            user_id
        )

    assert len(result) == 2
    assert {t["sender_id"] for t in result} == {"scan_s1", "scan_s2"}
