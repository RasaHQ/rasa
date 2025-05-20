import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import pytest
from dotenv import load_dotenv

from rasa.core.tracker_stores.mongo_tracker_store import MongoTrackerStore

if TYPE_CHECKING:
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker, Event


MONGODB_PARENT_PATH_NAME = (
    "tests_deployment/integration_tests_tracker_stores/" "mongo_db_tracker_store"
)


@pytest.fixture
def mongodb_credentials() -> Tuple[str, str, str]:
    """Get the MongoDB credentials from the .env file.

    For local testing, you should create a .env file in the path
    tests_deployment/integration_tests_tracker_stores.
    """
    load_dotenv(Path(f"{MONGODB_PARENT_PATH_NAME}/.env"))

    db_name = os.getenv("DB_NAME")
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    return db_name, username, password


def get_mongodb_tls_host_uri() -> str:
    """Prepare the MongoDB URI with TLS for the tests.

    You must generate the TLS certificates and keys for the MongoDB server
    you will run as a Docker container in the expected paths.
    """
    tls_ca_file = f"{MONGODB_PARENT_PATH_NAME}/tls/ca.crt"
    tls_certificate_key_file = f"{MONGODB_PARENT_PATH_NAME}/tls/mongodb.pem"

    return f"mongodb://localhost:27017/?tls=true&tlsCAFile={tls_ca_file}&tlsCertificateKeyFile={tls_certificate_key_file}"


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
def test_mongo_tracker_store_login(
    domain: "Domain",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Verify that the mongo client can connect to the mongodb server."""
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    assert mongo_tracker_store.client is not None

    assert db_name in mongo_tracker_store.client.list_database_names()


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_retrieve_full_tracker(
    domain: "Domain",
    tracker_with_restarted_event: "DialogueStateTracker",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Verify that the MongoTrackerStore can retrieve a full tracker."""
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    sender_id = tracker_with_restarted_event.sender_id

    await mongo_tracker_store.save(tracker_with_restarted_event)

    tracker = await mongo_tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_retrieve(
    domain: "Domain",
    tracker_with_restarted_event: "DialogueStateTracker",
    events_after_restart: List["Event"],
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    sender_id = tracker_with_restarted_event.sender_id
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )

    await mongo_tracker_store.save(tracker_with_restarted_event)

    tracker = await mongo_tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the MongoTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart[1:]


@pytest.mark.parametrize(
    "host_uri", ["mongodb://localhost:27000", get_mongodb_tls_host_uri()]
)
async def test_mongo_tracker_store_delete(
    domain: "Domain",
    tracker_with_restarted_event: "DialogueStateTracker",
    mongodb_credentials: Tuple[str, str, str],
    host_uri: str,
) -> None:
    """Verify that the MongoTrackerStore can delete a tracker."""
    # Given
    db_name, username, password = mongodb_credentials

    mongo_tracker_store = MongoTrackerStore(
        domain,
        host=host_uri,
        db=db_name,
        username=username,
        password=password,
        auth_source=db_name,
    )
    sender_id = tracker_with_restarted_event.sender_id
    await mongo_tracker_store.save(tracker_with_restarted_event)

    # When
    await mongo_tracker_store.delete(sender_id)

    # Then
    tracker = await mongo_tracker_store.retrieve(sender_id)
    assert tracker is None
