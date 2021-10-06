from __future__ import annotations
import logging

import rasa.utils.common
import rasa.core.actions.action
from typing import Dict, Text, Any

from rasa.core.channels.channel import (
    OutputChannel,
    CollectingOutputChannel,
)
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
    """Adds `SlotSet` events to DialogueStateTracker."""

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
        tracker: DialogueStateTracker,
        domain: Domain,
        nlg: NaturalLanguageGenerator,
        endpoint_config: EndpointConfig,
        output_channel: OutputChannel = CollectingOutputChannel(),
    ) -> DialogueStateTracker:
        """Adds slot extraction to the tracker.

        Args:
            tracker: The tracker to which slot extractions should be added to
            domain: The domain of the model.
            nlg: Indicates which natural language generator will be used.
            endpoint_config: The endpoint configuration.
            output_channel: The output channel which can be used to send messages.

        Returns:
            The original tracker updated with events created from the slot extractions.
        """
        action_extract_slots = rasa.core.actions.action.action_for_name_or_text(
            ACTION_EXTRACT_SLOTS, domain, endpoint_config,
        )
        extraction_events = rasa.utils.common.run_in_loop(
            action_extract_slots.run(output_channel, nlg, tracker, domain)
        )
        tracker.update_with_events(extraction_events, domain)

        events_as_str = "\n".join([str(e) for e in extraction_events])
        logger.debug(
            f"Default action '{ACTION_EXTRACT_SLOTS}' was executed, "
            f"resulting in {len(extraction_events)} events: {events_as_str}"
        )

        return tracker
