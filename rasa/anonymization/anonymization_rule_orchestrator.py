import copy
import json
import logging
from typing import Any, Dict, List, Optional, Text

from rasa.core.brokers.broker import EventBroker
from rasa.core.brokers.kafka import KafkaEventBroker

from rasa.anonymization.anonymization_rule_executor import (
    AnonymizationRuleExecutor,
    AnonymizationRuleList,
)

logger = logging.getLogger(__name__)


class AnonymizationRuleOrchestrator:
    """Orchestrates the execution of anonymization rules."""

    def __init__(
        self,
        event_broker: Optional[EventBroker],
        anonymization_rule_list: AnonymizationRuleList,
    ):
        """Initializes the orchestrator."""
        self.event_broker = (
            event_broker if isinstance(event_broker, KafkaEventBroker) else None
        )
        self.anonymization_rule_executor = AnonymizationRuleExecutor(
            anonymization_rule_list
        )

    def anonymize_event(self, event: Dict[Text, Any]) -> Dict[Text, Any]:
        """Anonymizes one of the supported event types.

        The supported event types: user, bot, slot, entities.
        """
        event_copy = copy.deepcopy(event)
        event_type = event["event"]

        if event_type == "user" or event_type == "bot":
            if event["text"] is None:
                return event_copy

            anonymized_text = self.anonymization_rule_executor.run(event["text"])
            event_copy["text"] = anonymized_text

            if "parse_data" in event_copy:
                event_copy["parse_data"]["text"] = anonymized_text

                entities = event_copy["parse_data"]["entities"]

                if entities:
                    anonymized_entities = self._anonymize_entities(entities)
                    event_copy["parse_data"]["entities"] = anonymized_entities

        elif event_type == "slot":
            slot_value = event["value"]

            if slot_value is None:
                return event_copy
            elif not isinstance(slot_value, str):
                slot_value = json.dumps(slot_value)

            anonymized_value = self.anonymization_rule_executor.run(slot_value)
            event_copy["value"] = anonymized_value

        elif event_type == "entities":
            entities = event_copy["entities"]
            anonymized_entities = self._anonymize_entities(entities)

            event_copy["entities"] = anonymized_entities
        else:
            logger.debug(f"Unsupported event type for anonymization: {event_type}.")

        return event_copy

    def publish_event(
        self, anonymized_event: Dict[Text, Any], is_anonymized: bool
    ) -> None:
        """Publishes the anonymized event to the event broker."""
        if self.event_broker is None:
            return None

        # this assumes that the event broker's topic attribute
        # is set to the correct anonymization topic
        if is_anonymized:
            event_type = anonymized_event["event"]
            sender_id = anonymized_event["sender_id"]
            logger.debug(
                f"Publishing event of type '{event_type}' from "
                f"sender ID '{sender_id}' to "
                f"Kafka topic '{self.event_broker.topic}'. "
                f"The anonymization pipeline used rule named "
                f"'{self.anonymization_rule_executor.anonymization_rule_list.id}'."
            )

        self.event_broker.publish(anonymized_event)

    def _anonymize_entities(
        self, entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """Anonymizes entities."""
        anonymized_entities = []

        for entity in entities:
            entity_value = entity.get("value")

            if entity_value is not None:
                value = self.anonymization_rule_executor.run(str(entity_value))
                entity["value"] = value
                anonymized_entities.append(entity)

        return anonymized_entities

    def anonymize_log_message(self, message: Text) -> Any:
        """Anonymizes log messages."""
        if self.event_broker:
            return None
        return self.anonymization_rule_executor.run(message)
