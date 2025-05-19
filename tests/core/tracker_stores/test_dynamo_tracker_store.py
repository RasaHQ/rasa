import uuid

import pytest
from moto import mock_aws
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.constants import ENV_SANIC_WORKERS
from rasa.core.tracker_stores.dynamo_tracker_store import DynamoTrackerStore
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import EndpointConfig
from tests.core.tracker_stores.conftest import get_or_create_tracker_store
from tests.utilities import filter_logs


# noinspection PyPep8Naming
@mock_aws
def test_dynamo_get_or_create(test_domain: Domain) -> None:
    get_or_create_tracker_store(DynamoTrackerStore(test_domain))


@mock_aws
async def test_dynamo_tracker_floats(test_domain: Domain) -> None:
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


@mock_aws
def test_dynamo_tracker_create_table_multiple_sanic_workers_error(
    test_domain: Domain,
    monkeypatch: MonkeyPatch,
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
