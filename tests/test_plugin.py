import warnings
from typing import Any, List, Optional, Text
from unittest.mock import MagicMock

import pytest
from pluggy import PluginManager
from pytest import MonkeyPatch

from rasa.core.brokers.broker import EventBroker
from rasa.core.tracker_store import TrackerStore
from rasa.plugin import plugin_manager
from rasa.shared.core.domain import Domain
from rasa.utils.endpoints import EndpointConfig


def test_plugin_manager() -> None:
    manager = plugin_manager()
    assert isinstance(manager, PluginManager)

    manager_2 = plugin_manager()
    assert manager_2 == manager


@pytest.mark.parametrize("event_broker", [None, MagicMock()])
def test_plugin_create_tracker_store(
    event_broker: Optional[EventBroker],
    monkeypatch: MonkeyPatch,
) -> None:
    manager = plugin_manager()
    monkeypatch.setattr(
        manager.hook, "create_tracker_store", MagicMock(return_value=None)
    )

    endpoint_config = EndpointConfig()
    domain = Domain.empty()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TrackerStore.create(
            obj=endpoint_config, domain=domain, event_broker=event_broker
        )

    manager.hook.create_tracker_store.assert_called_once_with(
        endpoint_config=endpoint_config, domain=domain, event_broker=event_broker
    )


@pytest.mark.parametrize("endpoints_file", [None, "test_endpoints.yml"])
def test_plugin_read_anonymization_rules(
    endpoints_file: Optional[Text],
    monkeypatch: MonkeyPatch,
) -> None:
    manager = plugin_manager()
    monkeypatch.setattr(
        manager.hook, "read_anonymization_rules", MagicMock(return_value=[])
    )

    manager.hook.read_anonymization_rules(endpoints_file=endpoints_file)
    manager.hook.read_anonymization_rules.assert_called_once_with(
        endpoints_file=endpoints_file
    )


@pytest.mark.parametrize(
    "anonymization_rules, event_broker_config",
    [
        (None, None),
        (None, EndpointConfig()),
        ([], None),
        ([], EndpointConfig()),
    ],
)
def test_plugin_create_anonymization_pipeline(
    monkeypatch: MonkeyPatch,
    anonymization_rules: Optional[List[Any]],
    event_broker_config: Optional[EndpointConfig],
) -> None:
    manager = plugin_manager()
    monkeypatch.setattr(
        manager.hook, "create_anonymization_pipeline", MagicMock(return_value=None)
    )

    manager.hook.create_anonymization_pipeline(
        anonymization_rules=anonymization_rules,
        event_broker_config=event_broker_config,
    )
    manager.hook.create_anonymization_pipeline.assert_called_once_with(
        anonymization_rules=anonymization_rules,
        event_broker_config=event_broker_config,
    )
