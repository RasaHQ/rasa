from typing import TYPE_CHECKING, List, Optional

import structlog

from rasa.core.brokers.broker import EventBroker
from rasa.utils.endpoints import EndpointConfig

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

structlogger = structlog.get_logger(__name__)


async def create_event_brokers(
    event_broker_endpoint: Optional[EndpointConfig],
    event_loop: Optional["AbstractEventLoop"] = None,
) -> List[EventBroker]:
    """Create EventBroker objects for each anonymization topic or queue."""
    if event_broker_endpoint is None or event_broker_endpoint.type is None:
        structlogger.debug(
            "rasa.privacy_filtering.create_event_broker.no_event_broker_type",
        )
        return []

    if event_broker_endpoint.type == "kafka":
        event_collection = event_broker_endpoint.kwargs.get("anonymization_topics", [])
        event_collection_type = "topic"
    elif event_broker_endpoint.type == "pika":
        event_collection = event_broker_endpoint.kwargs.get("anonymization_queues", [])
        event_collection_type = "queues"
    else:
        structlogger.debug(
            "rasa.privacy_filtering.create_event_broker.unsupported_event_broker_type",
            event_broker_type=event_broker_endpoint.type,
        )
        return []

    if not event_collection:
        structlogger.debug(
            f"rasa.privacy_filtering.create_event_broker.no_anonymization_{event_collection_type}",
            event_collection_type=event_collection_type,
        )
        return []

    return await _create_event_brokers(
        event_broker_endpoint, event_collection, event_collection_type, event_loop
    )


async def _create_event_brokers(
    event_broker_endpoint: EndpointConfig,
    event_collection: List[str],
    event_collection_type: str,
    event_loop: Optional["AbstractEventLoop"] = None,
) -> List[EventBroker]:
    """Create event brokers."""
    event_brokers = []
    for item in event_collection:
        event_broker_endpoint.kwargs[event_collection_type] = (
            item if event_collection_type == "topic" else [item]
        )
        structlogger.debug(
            "rasa.privacy_filtering.create_event_broker",
            event_info=f"Setting anonymized event streaming to '{item}'.",
        )

        event_broker = await EventBroker.create(event_broker_endpoint, event_loop)
        if event_broker is None:
            structlogger.debug(
                "rasa.privacy_filtering.create_event_broker.no_event_broker_created",
                event_info=f"No event broker created for publishing to '{item}'.",
            )
            continue

        event_brokers.append(event_broker)

    return event_brokers
