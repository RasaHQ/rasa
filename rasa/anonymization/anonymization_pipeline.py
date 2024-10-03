from __future__ import annotations

import copy
import json
import logging
import queue
from typing import Any, Dict, List, Optional, Text

from apscheduler.schedulers.background import BackgroundScheduler
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.shared.core.events import Event
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config

from rasa.anonymization.anonymisation_rule_yaml_reader import (
    AnonymizationRulesYamlReader,
)
from rasa.anonymization.anonymization_rule_executor import AnonymizationRuleList
from rasa.anonymization.anonymization_rule_orchestrator import (
    AnonymizationRuleOrchestrator,
)
from rasa.utils.singleton import Singleton

logger = logging.getLogger(__name__)


class AnonymizationPipeline:
    def run(self, event: Dict[Text, Any]) -> None:
        """Run the anonymization pipeline on the given event.

        Args:
            event: The event to anonymize
        """
        ...

    def log_run(self, data: Any) -> Any:
        """Anonymize the log data.

        Args:
            data: log data to anonymize

        Returns:
            Anonymized log data
        """
        ...

    def stop(self) -> None:
        """Stop the anonymization pipeline."""
        ...


class BackgroundAnonymizationPipeline(AnonymizationPipeline):
    EVENT_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS = 2.0

    def __init__(self, anonymization_pipeline: AnonymizationPipeline):
        self.anonymization_pipeline = anonymization_pipeline

        # Order of the initialisation is important
        # The event queue must be created before the scheduler
        # The can_consume_event_queue must be set to True before the scheduler starts
        self.event_queue: queue.Queue = queue.Queue()

        # This flag is used to stop the scheduler
        self.can_consume_from_event_queue = True
        self.event_anonymisation_scheduler = BackgroundScheduler()
        self.event_anonymisation_scheduler.add_job(
            self._consumer_queue, max_instances=1
        )
        self.event_anonymisation_scheduler.start()

    def stop(self) -> None:
        logger.debug("Shutting down the anonymization pipeline...")
        self.can_consume_from_event_queue = False
        self.event_anonymisation_scheduler.shutdown()

    def run(self, event: Dict[Text, Any]) -> None:
        self.event_queue.put(event)

    def log_run(self, data: Any) -> Any:
        return self.anonymization_pipeline.log_run(data)

    def _consumer_queue(self) -> None:
        while self.can_consume_from_event_queue:
            try:
                # Wait for 2 seconds for an event to be added to the queue
                # If no event is added to the queue, continue
                # This is done to avoid the scheduler to be stuck in the while loop
                # when we want to stop the scheduler
                event = self.event_queue.get(
                    timeout=self.EVENT_QUEUE_PROCESSING_TIMEOUT_IN_SECONDS
                )
                self.anonymization_pipeline.run(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue


class SyncAnonymizationPipeline(AnonymizationPipeline):
    """Pipeline for anonymizing events."""

    def __init__(self, orchestrators: List[AnonymizationRuleOrchestrator]) -> None:
        """Initializes the pipeline."""
        self.orchestrators = orchestrators

    def stop(self) -> None:
        pass

    def run(self, event: Dict[Text, Any]) -> None:
        """Runs the anonymization pipeline."""
        logger.debug("Running the anonymization pipeline for event...")

        for orchestrator in self.orchestrators:
            anonymized_event = orchestrator.anonymize_event(event)
            is_anonymized = True if event != anonymized_event else False
            orchestrator.publish_event(anonymized_event, is_anonymized)

    def log_run(self, data: Any) -> Any:
        """Runs the anonymization pipeline for logging."""
        logger.debug("Running the anonymization pipeline for logs....")

        anonymized_data = None

        # this is to make sure that the original data is not modified
        data_copy = copy.deepcopy(data)

        for orchestrator in self.orchestrators:
            # orchestrator for anonymizing logs has its event broker set to None
            if isinstance(orchestrator.event_broker, KafkaEventBroker):
                continue

            if isinstance(data_copy, str):
                anonymized_data = orchestrator.anonymize_log_message(data_copy)

            elif isinstance(data_copy, list):
                anonymized_data = [
                    Event.from_parameters(orchestrator.anonymize_event(item.as_dict()))
                    if isinstance(item, Event)
                    else orchestrator.anonymize_log_message(str(item["value"]))
                    if isinstance(item, dict) and item.get("value") is not None
                    else item
                    for item in data_copy
                ]

            elif isinstance(data_copy, dict) and "event" in data_copy:
                anonymized_data = orchestrator.anonymize_event(data_copy)

            elif isinstance(data_copy, dict):
                anonymized_data = {}

                for key, value in data_copy.items():
                    try:
                        serialized_value = json.dumps(value)
                    except TypeError as error:
                        logger.error(
                            f"Failed to serialize value of type '{type(value)}' "
                            f"for key '{key}' before anonymization. "
                            f"Encountered error: {error}. "
                            f"Setting value to None."
                        )
                        serialized_value = None

                    anonymized_data[key] = (
                        orchestrator.anonymize_log_message(serialized_value)
                        if serialized_value is not None
                        else None
                    )

            else:
                logger.debug("Unsupported data type for logging anonymization.")

            if anonymized_data:
                return anonymized_data
        return data


class AnonymizationPipelineProvider(metaclass=Singleton):
    """Represents a provider for anonymization pipeline."""

    anonymization_pipeline: Optional[AnonymizationPipeline] = None

    def register_anonymization_pipeline(self, pipeline: AnonymizationPipeline) -> None:
        """Register an anonymization pipeline.

        Args:
            pipeline: The anonymization pipeline to register.
        """
        self.anonymization_pipeline = pipeline

    def get_anonymization_pipeline(self) -> Optional[AnonymizationPipeline]:
        """Get the anonymization pipeline.

        Returns:
            The anonymization pipeline.
        """
        return self.anonymization_pipeline


def load_anonymization_pipeline(endpoints_file: Optional[Text]) -> None:
    """Creates an anonymization pipeline."""
    yaml_reader = AnonymizationRulesYamlReader(endpoints_filename=endpoints_file)
    anonymization_rules = yaml_reader.read_anonymization_rules()

    if anonymization_rules is None:
        return None

    event_broker_config = read_endpoint_config(
        endpoints_file, endpoint_type="event_broker"
    )
    logging_config = read_endpoint_config(endpoints_file, endpoint_type="logger")

    orchestrators = _load_orchestrators(
        logging_config, event_broker_config, anonymization_rules
    )

    if not orchestrators:
        return None

    pipeline = SyncAnonymizationPipeline(orchestrators)
    async_anonymization_pipeline = BackgroundAnonymizationPipeline(pipeline)
    provider = AnonymizationPipelineProvider()
    provider.register_anonymization_pipeline(async_anonymization_pipeline)

    return None


def create_event_broker(
    topic_name: Text, event_broker_config: EndpointConfig
) -> Optional[KafkaEventBroker]:
    """Create a KafkaEventBroker object.

    Returns None if the event broker config is not of type 'kafka'.
    """
    if event_broker_config.type != "kafka":
        logger.warning(
            f"Unsupported event broker config provided. "
            f"Expected type 'kafka' but got "
            f"'{event_broker_config.type}'. "
            f"Setting event broker to None."
        )
        event_broker = None
    else:
        logger.debug(f"Setting topic to '{topic_name}'.")
        event_broker_config.kwargs["topic"] = topic_name
        event_broker = KafkaEventBroker(
            event_broker_config.url, **event_broker_config.kwargs
        )

    return event_broker


def _load_orchestrators(
    logging_config: EndpointConfig,
    event_broker_config: EndpointConfig,
    anonymization_rules: List[AnonymizationRuleList],
) -> List[AnonymizationRuleOrchestrator]:
    orchestrators = []

    if logging_config:
        formatter = logging_config.kwargs.get("formatter", "")
        logging_rule = formatter.get("anonymization_rules")

        for rule_list in anonymization_rules:
            if rule_list.id == logging_rule:
                orchestrators.append(AnonymizationRuleOrchestrator(None, rule_list))

    if event_broker_config:
        anonymization_topics = event_broker_config.kwargs.get(
            "anonymization_topics", []
        )
        visited_topics = []

        for rule_list in anonymization_rules:
            for topic in anonymization_topics:
                topic_name = topic.get("name")

                if topic_name in visited_topics:
                    continue

                if rule_list.id == topic.get("anonymization_rules"):
                    event_broker = create_event_broker(topic_name, event_broker_config)

                    orchestrators.append(
                        AnonymizationRuleOrchestrator(event_broker, rule_list)
                    )

                    visited_topics.append(topic_name)
    return orchestrators
