from pathlib import Path
from typing import Any, Dict, Optional, Text, Tuple
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from rasa.core.tracker_store import AwaitableTrackerStore, MongoTrackerStore
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

from rasa.core.auth_retry_tracker_store import AuthRetryTrackerStore
from rasa.core.secrets_manager.vault import VaultSecretsManager, VaultTokenManager


class MockedMongoTrackerStore(MongoTrackerStore):
    """In-memory mocked version of `MongoTrackerStore`."""

    def __init__(
        self,
        _domain: Domain,
        username: Optional[Text] = None,
        password: Optional[Text] = None,
    ) -> None:
        from mongomock import MongoClient

        self.db: Any = MongoClient().rasa
        self.collection = "conversations"
        self.username = username
        self.password = password

        # Skip `MongoTrackerStore` constructor to avoid that actual Mongo connection
        # is created.
        super(MongoTrackerStore, self).__init__(_domain, None)


@pytest.fixture
def moodbot_domain() -> Domain:
    domain_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "test_domains"
        / "auth_retry_domain.yml"
    )
    return Domain.load(domain_path)


@pytest.fixture
def credentials() -> Tuple[Text, Text]:
    return "myusername", "mypassword"


@pytest.fixture
def sender_id() -> Text:
    return "integration_test_auth_retry"


@pytest.fixture
def tracker(sender_id: Text) -> DialogueStateTracker:
    events = [
        UserUttered("hello", {"name": "greet"}),
        ActionExecuted("utter_greet"),
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id=sender_id, evts=events, slots=[]
    )
    return tracker


@pytest.fixture
def vault_secrets_manager(
    credentials: Tuple[Text, Text],
    monkeypatch: MonkeyPatch,
) -> VaultSecretsManager:
    def mock_start(self: Any) -> None:
        pass

    monkeypatch.setattr(VaultTokenManager, "start", mock_start)

    def mock_load_secrets(self: Any) -> Dict[Text, Text]:
        return {"username": credentials[0], "password": credentials[1]}

    monkeypatch.setattr(VaultSecretsManager, "load_secrets", mock_load_secrets)

    return VaultSecretsManager(
        # deepcode ignore HardcodedNonCryptoSecret/test: Test secret
        host="localhost:8200",
        token="myroot",
        # deepcode ignore HardcodedNonCryptoSecret/test: Test credential
        secrets_path="rasa-secrets",
    )


@pytest.fixture
def mock_create_tracker_store(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_create_tracker_store = MagicMock()
    return _mock_create_tracker_store


@pytest.fixture
def set_mock_create_tracker_store(
    mock_create_tracker_store: MagicMock, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rasa.core.auth_retry_tracker_store.create_tracker_store",
        mock_create_tracker_store,
    )


async def test_auth_retry_tracker_store_save_retrieve_keys(
    moodbot_domain: Domain,
    sender_id: Text,
    tracker: DialogueStateTracker,
    mock_create_tracker_store: MagicMock,
    set_mock_create_tracker_store: None,
) -> None:
    tracker_store = AwaitableTrackerStore(MockedMongoTrackerStore(moodbot_domain))
    mock_create_tracker_store.return_value = tracker_store
    auth_retry_tracker_store = AuthRetryTrackerStore(
        domain=moodbot_domain, endpoint_config=EndpointConfig(), retries=1
    )

    # check there is no tracker stored with this sender_id
    non_existing_tracker = await auth_retry_tracker_store.retrieve(sender_id=sender_id)
    assert non_existing_tracker is None

    # save tracker
    await auth_retry_tracker_store.save(tracker)

    # retrieve tracker
    tracker_from_auth_retry = await auth_retry_tracker_store.retrieve(
        sender_id=sender_id
    )
    assert tracker_from_auth_retry == tracker

    tracker_from_underlying_store = await tracker_store.retrieve(sender_id=sender_id)
    assert tracker_from_auth_retry == tracker_from_underlying_store

    keys = await auth_retry_tracker_store.keys()
    assert sender_id in keys
