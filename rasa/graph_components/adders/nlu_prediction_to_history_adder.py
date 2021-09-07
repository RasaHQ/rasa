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
        predictions: List["Message"],
        tracker: "DialogueStateTracker",
        domain: Domain,
        original_message: UserMessage,
    ) -> "DialogueStateTracker":
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
            user_utterance = UserUttered(
                message.data.get(TEXT),
                message.data.get(INTENT),
                message.data.get(ENTITIES),
                message.data,
                None,
                original_message.input_channel,
                message.data.get("message_id"),
                message.data.get("metadata"),
            )
            tracker.update(user_utterance, domain)

        logger.debug(
            f"Logged {len(predictions)} UserUtterance(s) - \
                tracker now has {len(tracker.events)} events."
        )

        return tracker
