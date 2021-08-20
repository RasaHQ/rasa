from __future__ import annotations
from typing import Dict, Text, Any, Optional, List

from rasa.core.channels.channel import UserMessage

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES
from rasa.shared.nlu.training_data.message import Message


class NLUMessageConverter(GraphComponent):
    """Converts the user message into a NLU Message object."""

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Creates converter from config."""
        self._config = config

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default configuration (see parent class for full docstring)."""
        return {
            "remove_duplicates": True,
            "unique_last_num_states": None,
            "augmentation_factor": 50,
            "tracker_limit": None,
            "use_story_concatenation": True,
            "debug_plots": False,
        }

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUMessageConverter:
        """Creates component (see parent class for full docstring)."""
        return cls(config)

    @staticmethod
    def convert_user_message(message: Optional[UserMessage]) -> List[Optional[Message]]:
        """Converts user message into Message object.

        Returns:
            List containing only one instance of Message.
            Else empty list if user message is None.
        """
        if message:
            data = dict()
            data[TEXT] = message.text
            data["input_channel"] = message.input_channel
            data["message_id"] = message.message_id
            data["metadata"] = message.metadata

            if message.parse_data:
                data[INTENT] = message.parse_data[INTENT]
                data[ENTITIES] = message.parse_data[ENTITIES]
                data["parse_data"] = message.parse_data

            return [Message(data=data)]

        return []
