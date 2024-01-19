import pytest
from rasa.core.brokers.kafka import KafkaEventBroker

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationRule,
    AnonymizationRuleList,
)
from rasa.anonymization.anonymization_rule_orchestrator import (
    AnonymizationRuleOrchestrator,
)


@pytest.fixture(scope="session")
def anonymization_rule_list() -> AnonymizationRuleList:
    rule_list = AnonymizationRuleList(
        id="test_events",
        rule_list=[
            AnonymizationRule(
                entity_name="PERSON",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="IBAN_CODE",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="CREDIT_CARD",
                substitution="faker",
            ),
            AnonymizationRule(
                entity_name="LOCATION",
                substitution="faker",
            ),
        ],
    )

    return rule_list


@pytest.fixture(scope="session")
def anonymization_rule_orchestrator(
    anonymization_rule_list: AnonymizationRuleList,
) -> AnonymizationRuleOrchestrator:
    event_broker = KafkaEventBroker(url="localhost", topic="topic1")

    orchestrator = AnonymizationRuleOrchestrator(
        event_broker=event_broker, anonymization_rule_list=anonymization_rule_list
    )

    return orchestrator


@pytest.fixture(scope="session")
def anonymization_rule_orchestrator_no_event_broker(
    anonymization_rule_list: AnonymizationRuleList,
) -> AnonymizationRuleOrchestrator:
    orchestrator = AnonymizationRuleOrchestrator(
        event_broker=None, anonymization_rule_list=anonymization_rule_list
    )

    return orchestrator
