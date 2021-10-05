from __future__ import annotations
import logging

import rasa.core.actions.action
from typing import Dict, Text, Any, Optional

from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.utils.endpoints import EndpointConfig

from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain
from rasa.shared.core.constants import ACTION_EXTRACT_SLOTS

logger = logging.getLogger(__name__)


class SlotExtractionToHistoryAdder(GraphComponent):
    """Adds slot extractions to DialogStateTracker."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SlotExtractionToHistoryAdder:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def add(
        self,
        endpoint_config: Optional[EndpointConfig],
        tracker: DialogueStateTracker,
        domain: Domain,
        nlg: NaturalLanguageGenerator,
        message: UserMessage,
    ) -> DialogueStateTracker:
        action_extract_slots = rasa.core.actions.action.action_for_name_or_text(
            ACTION_EXTRACT_SLOTS, domain, endpoint_config,
        )
        output_channel = (
            message.output_channel
            if message.output_channel
            else CollectingOutputChannel()
        )
        extraction_events = await action_extract_slots.run(
            output_channel, nlg, tracker, domain
        )
        tracker.update_with_events(extraction_events, domain)

        events_as_str = "\n".join([str(e) for e in extraction_events])
        logger.debug(
            f"Default action '{ACTION_EXTRACT_SLOTS}' was executed, "
            f"resulting in {len(extraction_events)} events: {events_as_str}"
        )

        return tracker
