import logging
from typing import Any, Dict, List, Optional, Text

import rasa.utils.common as common_utils
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_END,
)
from rasa.nlu.training_data import Message
from rasa.nlu.extractors.extractor import EntityExtractor

logger = logging.getLogger(__name__)


class LookupEntityExtractor(EntityExtractor):
    """
    Searches for entities in the user's message from a list of examples.
    Required Parameters:
    @lookup -> dict
    """

    defaults = {
        # lower case the entity value from the lookup file and
        # user message while comparing them
        "lowercase": True
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(LookupEntityExtractor, self).__init__(component_config)

        self.lowercase = self.component_config["lowercase"]
        self.lookup = {}

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        if not training_data.lookup_tables:
            common_utils.raise_warning("No lookup tables defined.")
            return

        for table in training_data.lookup_tables:
            self.lookup[table["name"]] = table["elements"]

    def _parse_message(self, user_message: Text) -> List[Dict[Text, Any]]:
        """Parse the given user message and extract the entities."""
        entities = []

        for entity, elements in self.lookup.items():
            entities += self._extract_entities(user_message, entity, elements)

        return entities

    def _extract_entities(
        self, user_message: Text, entity: Text, elements: List[Text]
    ) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        entities = []
        _user_message = user_message
        if self.lowercase:
            _user_message = _user_message.lower()

        for example in elements:
            example = example.strip()
            if self.lowercase:
                example = example.lower()

            if example in _user_message:
                start_index = _user_message.index(example)
                end_index = start_index + len(example)
                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: entity,
                        ENTITY_ATTRIBUTE_START: start_index,
                        ENTITY_ATTRIBUTE_END: end_index,
                        ENTITY_ATTRIBUTE_VALUE: user_message[start_index:end_index],
                    }
                )

        return entities

    def process(self, message: Message, **kwargs: Any) -> None:
        """Retrieve the text message, parse the entities."""

        extracted_entities = self._parse_message(message.text)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )
