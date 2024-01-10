import copy
import logging
import time
from typing import Any, Dict, List, Text
from unittest.mock import MagicMock, Mock

import pytest
from pytest import LogCaptureFixture, MonkeyPatch
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.shared.core.events import BotUttered, EntitiesAdded, SlotSet, UserUttered
from rasa.utils.endpoints import EndpointConfig

from rasa.anonymization.anonymization_pipeline import (
    AnonymizationPipelineProvider,
    BackgroundAnonymizationPipeline,
    SyncAnonymizationPipeline,
    create_event_broker,
)
from rasa.anonymization.anonymization_rule_orchestrator import (
    AnonymizationRuleOrchestrator,
)


def test_anonymization_pipeline_run_valid_events(
    caplog: LogCaptureFixture,
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests the anonymization pipeline with valid events.

    Valid event types include `slot`, `entities`, `bot`, `user`.
    """
    events: List[Dict[Text, Any]] = [
        {
            "event": "user",
            "text": "My name is Julia.",
            "sender_id": 1234,
        },
        {
            "event": "bot",
            "text": "Money transfer to IBAN ES79 2100 0813 6101 2345 6789 "
            "was successfully completed.",
            "sender_id": 1234,
        },
        {
            "event": "entities",
            "entities": [{"entity": "credit_card", "value": "4916741327614057"}],
            "sender_id": 1234,
        },
        {"event": "slot", "value": "London", "sender_id": 1234},
    ]

    mock_publish = MagicMock()
    monkeypatch.setattr(
        anonymization_rule_orchestrator.event_broker, "publish", mock_publish
    )

    anonymization_pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator]
    )

    with caplog.at_level(logging.DEBUG):
        for event in events:
            anonymization_pipeline.run(event)

        pipeline_caplog_records = list(
            filter(
                lambda record: "Running the anonymization pipeline for event..."
                in record.message,
                caplog.records,
            )
        )

        assert len(pipeline_caplog_records) == len(events)
        assert mock_publish.call_count == len(events)


@pytest.mark.parametrize(
    "data",
    [
        "My name is Jane Doe.",
        [{"entity": "name", "value": "Kristin Cruz"}],
        {
            "event": "user",
            "text": "My name is Julia.",
            "sender_id": 1234,
        },
        {"city": "London"},
        [BotUttered("Your son's name Jamie Doe.")],
        [UserUttered("My name is Jane Doe.")],
        [SlotSet("name", "Jane Doe")],
        [
            EntitiesAdded(
                entities=[
                    {"entity": "city", "value": "London"},
                    {"entity": "name", "value": "Joanna Joe"},
                ],
                timestamp=None,
            ),
        ],
    ],
)
def test_anonymization_pipeline_log_run(
    caplog: LogCaptureFixture,
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
    data: Any,
) -> None:
    """Tests the anonymization pipeline with logging."""
    anonymization_pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator_no_event_broker]
    )
    original_data = copy.deepcopy(data)

    with caplog.at_level(logging.DEBUG):
        anonymized_data = anonymization_pipeline.log_run(data)
        text = "Running the anonymization pipeline for logs..."

        pipeline_caplog_records = list(
            filter(
                lambda record: text in record.message,
                caplog.records,
            )
        )

        assert len(pipeline_caplog_records) == 1
        assert anonymized_data != original_data


def test_create_event_broker_invalid(
    caplog: LogCaptureFixture,
) -> None:
    topic_name = "test_topic_1"

    kwargs = {"type": "pika", "username": "guest", "password": "guest"}
    event_broker_config = EndpointConfig(url="localhost", **kwargs)

    with caplog.at_level(logging.WARNING):
        event_broker = create_event_broker(topic_name, event_broker_config)
        msg = (
            "Unsupported event broker config provided. "
            "Expected type 'kafka' but got 'pika'. "
            "Setting event broker to None."
        )
        assert msg in caplog.text

    assert event_broker is None


def test_create_event_broker_valid(
    caplog: LogCaptureFixture,
) -> None:
    topic_name = "test_topic_2"

    kwargs = {"type": "kafka"}
    event_broker_config = EndpointConfig(url="localhost", **kwargs)

    with caplog.at_level(logging.DEBUG):
        event_broker = create_event_broker(topic_name, event_broker_config)
        msg = f"Setting topic to '{topic_name}'."
        assert msg in caplog.text

    assert isinstance(event_broker, KafkaEventBroker)
    assert event_broker.topic == topic_name


def test_anonymization_pipeline_provider_is_singleton(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    pipeline_provider_1 = AnonymizationPipelineProvider()
    pipeline_provider_2 = AnonymizationPipelineProvider()

    assert pipeline_provider_1 is pipeline_provider_2
    assert (
        pipeline_provider_1.anonymization_pipeline
        is pipeline_provider_2.anonymization_pipeline
    )


def test_anonymization_pipeline_provider(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    pipeline_provider = AnonymizationPipelineProvider()
    anonymization_pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator]
    )
    pipeline_provider.register_anonymization_pipeline(anonymization_pipeline)

    assert pipeline_provider.anonymization_pipeline == anonymization_pipeline
    assert pipeline_provider.get_anonymization_pipeline() == anonymization_pipeline


def test_anonymization_pipeline_does_not_anonymize_logs_when_not_enabled(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
) -> None:
    pipeline = SyncAnonymizationPipeline([anonymization_rule_orchestrator])
    data = [UserUttered("My name is Jane Doe.")]
    anonymized_data = pipeline.log_run(data)
    assert anonymized_data == data


def test_anonymization_pipeline_anonymize_logs_when_enabled(
    anonymization_rule_orchestrator: AnonymizationRuleOrchestrator,
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
) -> None:
    pipeline = SyncAnonymizationPipeline(
        [
            anonymization_rule_orchestrator,
            anonymization_rule_orchestrator_no_event_broker,
        ]
    )
    data = [
        UserUttered(
            "My name is Jane Doe.",
            parse_data={
                "intent": {},
                "entities": [{"entity": "name", "value": "Jane Doe"}],
            },
        )
    ]
    anonymized_data = pipeline.log_run(data)

    assert anonymized_data[0].text != data[0].text
    assert anonymized_data[0].parse_data["text"] != data[0].parse_data["text"]
    assert (
        anonymized_data[0].parse_data["entities"][0]["value"]
        != data[0].parse_data["entities"][0]["value"]
    )


@pytest.mark.parametrize(
    "data",
    [
        [{"entity": "name", "value": None}],
        {
            "event": "slot",
            "name": "city",
            "value": None,
        },
        [SlotSet("name", None)],
        [
            UserUttered(
                "I want to open a current account",
                parse_data={
                    "entities": [{"entity": "name", "value": None}],
                    "text": "I want to open a current account",
                },
            )
        ],
    ],
)
def test_anonymization_pipeline_anonymize_logs_with_none_values(
    data: Any,
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
) -> None:
    pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator_no_event_broker]
    )
    anonymized_data = pipeline.log_run(data)

    assert anonymized_data == data


def test_anonymization_pipeline_anonymize_logs_with_none_entities_event(
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
) -> None:
    data = [
        EntitiesAdded(
            entities=[
                {"entity": "name", "value": None},
            ],
            timestamp=None,
        ),
    ]
    pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator_no_event_broker]
    )
    anonymized_data = pipeline.log_run(data)

    assert anonymized_data[0].entities == []


def test_background_anonymization_pipeline_event_pii() -> None:
    mock_anonymization_pipeline = MagicMock()
    mock_anonymization_pipeline.run = MagicMock()

    ### Set the event processing timeout to a very low value to speed up the test
    BackgroundAnonymizationPipeline.EVENT_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS = 0.1
    background_anon_pipeline = BackgroundAnonymizationPipeline(
        mock_anonymization_pipeline
    )

    event = {"event": "user", "text": "My name is Julia."}
    background_anon_pipeline.run(event)

    ### We need to wait for the event to be processed in the background thread
    time.sleep(
        BackgroundAnonymizationPipeline.EVENT_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS
    )

    mock_anonymization_pipeline.run.assert_called_once_with(event)
    ### Stop the background thread
    background_anon_pipeline.stop()


def test_background_anonymization_pipeline_log_pii() -> None:
    mock_anonymization_pipeline = MagicMock()
    mock_anonymization_pipeline.log_run = MagicMock()

    background_anon_pipeline = BackgroundAnonymizationPipeline(
        mock_anonymization_pipeline
    )

    log = "My name is Julia."
    background_anon_pipeline.log_run(log)

    mock_anonymization_pipeline.log_run.assert_called_once_with(log)
    ### Stop the background thread
    background_anon_pipeline.stop()


def test_sync_anonymization_pipeline_casts_int_log_to_string(
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
) -> None:
    int_slot_value = 1
    data = [{"event": "slot", "name": "city", "value": int_slot_value}]
    pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator_no_event_broker]
    )
    anonymized_data = pipeline.log_run(data)
    assert anonymized_data == [str(int_slot_value)]


def test_sync_anonymization_pipeline_log_run_unserializable_dict_value(
    caplog: LogCaptureFixture,
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
) -> None:
    anonymization_pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator_no_event_broker]
    )
    unserializable_data = {"test": Mock()}

    with caplog.at_level(logging.ERROR):
        anonymized_data = anonymization_pipeline.log_run(unserializable_data)
        error_message = (
            "Failed to serialize value of type "
            "'<class 'unittest.mock.Mock'>' for key 'test' "
            "before anonymization. "
            "Encountered error: "
            "Object of type Mock is not JSON serializable. "
            "Setting value to None."
        )
        assert error_message in caplog.text
        assert anonymized_data == {"test": None}


class MockSerializable(dict):
    def __init__(self, value: Any) -> None:
        self.value = value
        super().__init__()


@pytest.mark.parametrize(
    "data",
    [
        {"test1": MockSerializable("test")},
        {"test2": [MockSerializable(1)]},
        {"test3": {"test4": MockSerializable(2)}},
    ],
)
def test_sync_anonymization_pipeline_log_run_serializable_dict_value(
    caplog: LogCaptureFixture,
    anonymization_rule_orchestrator_no_event_broker: AnonymizationRuleOrchestrator,
    data: Any,
) -> None:
    anonymization_pipeline = SyncAnonymizationPipeline(
        [anonymization_rule_orchestrator_no_event_broker]
    )

    with caplog.at_level(logging.ERROR):
        anonymized_data = anonymization_pipeline.log_run(data)

        # filter out ddtrace logs
        records = [record for record in caplog.records if "ddtrace" not in record.name]
        assert not records

    assert None not in anonymized_data.values()
