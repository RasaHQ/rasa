from __future__ import annotations
from typing import Dict, Text, Any, List

from rasa.core.channels.channel import UserMessage

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT, TEXT_TOKENS
from rasa.shared.nlu.training_data.message import Message


class NLUMessageConverter(GraphComponent):
    """Converts the user message into a NLU Message object."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUMessageConverter:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def convert_user_message(self, messages: List[UserMessage]) -> List[Message]:
        """Converts user message into Message object.

        Args:
            messages: The user messages which should be converted to be processed by
                the NLU components.

        Returns:
            List containing only one instance of Message.
            Else empty list if user message is None.
        """
        return [
            Message(
                data={
                    TEXT: message.text,
                    "message_id": message.message_id,
                    "metadata": message.metadata,
                },
                output_properties={TEXT_TOKENS},
            )
            for message in messages
        ]
