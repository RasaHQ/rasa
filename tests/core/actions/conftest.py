from unittest.mock import MagicMock

import pytest

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig


@pytest.fixture
def mock_endpoint() -> EndpointConfig:
    endpoint = MagicMock(spec=EndpointConfig)
    endpoint.url = "http://localhost:5055/webhook"
    endpoint.kwargs = {}
    return endpoint


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events("test", evts=[SlotSet("foo", "bar")])


@pytest.fixture
def domain() -> Domain:
    return Domain.from_dict({"responses": {}})
