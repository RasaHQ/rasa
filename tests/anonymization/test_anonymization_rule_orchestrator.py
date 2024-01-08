import json
import logging
from typing import Any, Dict, Text
from unittest.mock import MagicMock

import pytest
from pytest import LogCaptureFixture, MonkeyPatch
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.brokers.pika import PikaEventBroker

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationRule,
    AnonymizationRuleList,
)
from rasa.anonymization.anonymization_rule_orchestrator import (
    AnonymizationRuleOrchestrator,
)


def test_anonymization_rule_orchestrator_initialization_without_kafka() -> None:
    event_broker = PikaEventBroker(host="localhost", username="guest", password="guest")
    rule_list = AnonymizationRuleList(
        id="test",
        rule_list=[
            AnonymizationRule(
                entity_name="EMAIL_ADDRESS",
                substitution="mask",
            ),
        ],
    )

    orchestrator = AnonymizationRuleOrchestrator(
        event_broker=event_broker, anonymization_rule_list=rule_list
    )

    assert orchestrator is not None
    assert orchestrator.event_broker is None
    assert orchestrator.anonymization_rule_executor.anonymization_rule_list == rule_list


def test_anonymization_rule_orchestrator_initialization_with_kafka(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    anonymization_rule_list: AnonymizationRuleList,
) -> None:
    assert anonymization_rule_orchestrator is not None
    assert isinstance(anonymization_rule_orchestrator.event_broker, KafkaEventBroker)
    assert (
        anonymization_rule_orchestrator.anonymization_rule_executor.anonymization_rule_list
        == anonymization_rule_list
    )


def test_anonymization_rule_orchestrator_anonymize_user_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event = {"event": "user", "text": "My name is Julia."}
    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    assert anonymized_event.get("text") != event.get("text")


def test_anonymization_rule_orchestrator_anonymize_bot_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    original_iban = "ES79 2100 0813 6101 2345 6789"
    event = {
        "event": "bot",
        "text": f"Money transfer to IBAN {original_iban} was successfully completed.",
    }
    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    assert anonymized_event.get("text") != event.get("text")

    assert original_iban not in anonymized_event.get("text", "")
    assert original_iban.replace(" ", "") not in anonymized_event.get("text", "")


def test_anonymization_rule_orchestrator_anonymize_entities_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event = {
        "event": "entities",
        "entities": [{"entity": "credit_card", "value": "4916741327614057"}],
    }

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    anonymized_entity = anonymized_event.get("entities", [])[0].get("value")
    original_entity = event.get("entities", [])[0].get("value")  # type: ignore[attr-defined] # noqa: E501

    assert anonymized_entity != original_entity


def test_anonymization_rule_orchestrator_anonymize_slot_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event = {"event": "slot", "value": "London"}

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    assert anonymized_event.get("value") != event.get("value")


def test_anonymization_rule_orchestrator_publish_anonymized_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    event = {"event": "user", "text": "My name is *****.", "sender_id": 1}
    text = "Publishing event of type 'user' from sender ID '1' to Kafka topic 'topic1'."

    mock_publish = MagicMock()
    monkeypatch.setattr(
        anonymization_rule_orchestrator.event_broker, "publish", mock_publish
    )

    with caplog.at_level(logging.DEBUG):
        anonymization_rule_orchestrator.publish_event(event, True)

        assert text in caplog.text
        mock_publish.assert_called_once_with(event)


def test_anonymization_rule_orchestrator_publish_unanonymized_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests that there are no logs for unanonymised events."""
    event = {"event": "user", "text": "This is a beautiful.", "sender_id": 2}
    text = "Publishing event of type 'user' from sender ID '2' to Kafka topic 'topic1'."

    mock_publish = MagicMock()
    monkeypatch.setattr(
        anonymization_rule_orchestrator.event_broker, "publish", mock_publish
    )

    with caplog.at_level(logging.DEBUG):
        anonymization_rule_orchestrator.publish_event(event, False)

        assert text not in caplog.text
        mock_publish.assert_called_once_with(event)


@pytest.mark.parametrize(
    "event",
    [
        {"event": "action", "name": "action_listen"},
        {"event": "restart"},
        {"event": "session_started"},
        {
            "event": "reminder",
            "intent": "wake_up",
            "date_time": "2020-08-25T09:00:00.000+02:00",
        },
        {"event": "cancel_reminder"},
    ],
)
def test_anonymization_rule_orchestrator_does_not_anonymize_invalid_events(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
    event: Dict[Text, Any],
) -> None:
    event_type = event.get("event")

    with caplog.at_level(logging.DEBUG):
        event_copy = anonymization_rule_orchestrator.anonymize_event(event)

        assert f"Unsupported event type for anonymization: {event_type}." in caplog.text
        assert event_copy == event


def test_anonymization_rule_orchestrator_anonymizes_parse_data_in_user_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event: Dict[Text, Any] = {
        "event": "user",
        "text": "My name is Julia.",
        "parse_data": {
            "intent": {"name": "greet", "confidence": 0.9},
            "entities": [
                {"entity": "name", "value": "Julia", "confidence": 0.9},
            ],
            "text": "My name is Julia.",
        },
    }

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    anonymized_entities = anonymized_event.get("parse_data", {}).get("entities")
    original_entities = event.get("parse_data", {}).get("entities")

    assert anonymized_entities is not None
    assert len(anonymized_entities) == len(original_entities)

    for anonymized_entity, original_entity in zip(
        anonymized_entities, original_entities
    ):
        assert anonymized_entity.get("value") != original_entity.get("value")

    anonymized_text_in_parse_data = anonymized_event.get("parse_data", {}).get("text")
    original_text_in_parse_data = event.get("parse_data", {}).get("text")

    assert anonymized_text_in_parse_data != original_text_in_parse_data


def test_anonymization_rule_orchestrator_skip_none_entity_in_user_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event: Dict[Text, Any] = {
        "event": "user",
        "text": "My name is Julia.",
        "parse_data": {
            "intent": {"name": "greet", "confidence": 0.9},
            "entities": [
                {"entity": "name", "value": None},
            ],
            "text": "My name is Julia.",
        },
    }

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    anonymized_entities = anonymized_event.get("parse_data", {}).get("entities")
    assert anonymized_entities == []


def test_anonymization_rule_orchestrator_skip_none_entity_in_entities_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event: Dict[Text, Any] = {
        "event": "entities",
        "entities": [
            {"entity": "name", "value": None},
        ],
    }

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    anonymized_entities = anonymized_event.get("entities")
    assert anonymized_entities == []


def test_anonymization_rule_orchestrator_anonymize_log_message(
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
) -> None:
    name = "Julia"
    message = f"User says: My name is {name}."
    anonymized_message = (
        anonymization_rule_orchestrator_no_event_broker.anonymize_log_message(message)
    )

    assert anonymized_message != message
    assert name not in anonymized_message


def test_anonymization_rule_orchestrator_anonymize_bot_none_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    event = {
        "event": "bot",
        "text": None,
    }
    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    assert anonymized_event == event


@pytest.mark.parametrize("value", [1, 1.0, True, False, [], {}])
def test_anonymization_rule_orchestrator_anonymize_slot_event_non_string_value(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    value: Any,
) -> None:
    event = {"event": "slot", "value": value}

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    assert anonymized_event == {"event": "slot", "value": json.dumps(value)}


@pytest.mark.parametrize("value", [1, 1.0, True, False])
def test_anonymization_rule_orchestrator_non_string_entity_in_entities_event(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    value: Any,
) -> None:
    event: Dict[Text, Any] = {
        "event": "entities",
        "entities": [
            {"entity": "name", "value": value},
        ],
    }

    anonymized_event = anonymization_rule_orchestrator.anonymize_event(event)

    anonymized_entities = anonymized_event.get("entities")
    assert anonymized_entities == [{"entity": "name", "value": str(value)}]
