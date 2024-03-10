from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Text, List

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=False
)
class RegexMessageHandler(GraphComponent, EntityExtractorMixin):
    """Handles hardcoded NLU predictions from messages starting with a `/`."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RegexMessageHandler:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls()

    # TODO: Handle empty domain (NLU only training)
    def process(
        self, messages: List[Message], domain: Optional[Domain] = None
    ) -> List[Message]:
        """Adds hardcoded intents and entities for messages starting with '/'.

        Args:
            messages: The messages which should be handled.
            domain: If given the domain is used to check whether the intent, entities
                valid.

        Returns:
            The messages with potentially intent and entity prediction replaced
            in case the message started with a `/`.
        """
        return [
            YAMLStoryReader.unpack_regex_message(
                message, domain, entity_extractor_name=self.name
            )
            for message in messages
        ]
