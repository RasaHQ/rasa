from __future__ import annotations
from rasa.shared.core.events import UserUttered
from typing import Dict, Text, Any, Optional, List

from rasa.core.channels.channel import UserMessage

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain


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

    # Should this be predict since it's run during inference?
    def add(
        predictions: List["Message"],
        dialogue_tracker: "DialogueStateTracker",
        domain: Domain,
        original_message: UserMessage,
    ) -> "DialogueStateTracker":
        for message in predictions:
            # Need to find out where to get time from
            # Message contains time but it is an int, we want a float
            # What is domain for?
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
            user_utterance.apply_to(dialogue_tracker)

        return dialogue_tracker
