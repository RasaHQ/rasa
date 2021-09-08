from __future__ import annotations
import logging
from rasa.shared.core.events import UserUttered
from typing import Dict, Text, Any, List

from rasa.core.channels.channel import UserMessage

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)


class NLUPredictionToHistoryAdder(GraphComponent):
    """Adds NLU predictions to DialogueStateTracker."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUPredictionToHistoryAdder:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def add(
        self,
        predictions: List[Message],
        tracker: DialogueStateTracker,
        domain: Domain,
        original_message: UserMessage,
    ) -> DialogueStateTracker:
        """Adds NLU predictions to the tracker.

        Args:
            predictions: A list of NLU predictions wrapped as Messages
            tracker: The tracker the predictions should be attached to
            domain: The domain of the model.
            original_message: An original message from the user with
                extra metadata to annotate the predictions (e.g. channel)

        Returns:
            The original tracker updated with events created from the predictions
        """
        for message in predictions:
            user_event = UserUttered(
                message.data.get(TEXT),
                message.data.get(INTENT),
                message.data.get(ENTITIES),
                input_channel=original_message.input_channel,
                message_id=message.data.get("message_id"),
                metadata=original_message.metadata,
            )
            tracker.update(user_event, domain)

            if user_event.entities:
                # Log currently set slots
                slot_values = "\n".join(
                    [f"\t{s.name}: {s.value}" for s in tracker.slots.values()]
                )
                if slot_values.strip():
                    logger.debug(f"Current slot values: \n{slot_values}")

        logger.debug(
            f"Logged {len(predictions)} UserUtterance(s) - \
                tracker now has {len(tracker.events)} events."
        )

        return tracker
