import os
import logging
from typing import Any, Dict, List, Optional, Text

import rasa.utils.common as common_utils
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
        # lookup key for extracting lookup entities,
        # it contains the dictionary of lookup entity names
        # and their respective data files
        # example:
        # - name: LookupEntityExtractor
        #   lookup:
        #      city: /some/path/city.txt
        #      person: /some/other/path/person.txt
        "lookup": None,
        # lower case the entity value from the lookup file and
        # user message while comparing them
        "lowercase": True,
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(LookupEntityExtractor, self).__init__(component_config)

        if self.component_config["lookup"] is None:
            message = (
                "Can't extract any entities as the option 'lookup' is not set."
                "Please provide valid lookup entries for entities and their respective "
                "file paths."
            )
            raise ValueError(message)

        self.lookup = self.component_config["lookup"]
        self.lowercase = self.component_config["lowercase"]

        # check if the entities and respective file path exists
        for entity_type, file_path in self.lookup.items():
            self._validate_lookup_entry(entity_type, file_path)

        if not self.lookup:
            message = (
                "Can't extract any entities as the option 'lookup' just contains "
                "invalid entries. Please provide valid lookup entries for entities and "
                "their respective file paths."
            )
            raise ValueError(message)

    def _validate_lookup_entry(self, entity: Text, file_path: Text) -> None:
        if file_path is None:
            # remove the entity from the lookup dictionary,
            # if the file path is not provided
            self.lookup.pop(entity)
            message = (
                f"No file path for entity '{entity}' was given. "
                f"Please provide a valid file path."
            )
            common_utils.raise_warning(message)

        elif not os.path.isfile(file_path):
            # remove the entity from the lookup dictionary,
            # if the file path doesn't exist
            self.lookup.pop(entity)
            message = (
                f"The file path '{file_path}' for entity '{entity}' does not exist. "
                f"Please provide a valid file path."
            )
            common_utils.raise_warning(message)

    def _parse_message(self, user_message: Text) -> List[Dict[Text, Any]]:
        """Parse the given user message and extract the entities."""
        entities = []

        for entity, file_path in self.lookup.items():
            entities += self._extract_entities(user_message, entity, file_path)

        return entities

    def _extract_entities(
        self, user_message: Text, entity: Text, file_path: Text
    ) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        entities = []
        _user_message = user_message
        if self.lowercase:
            _user_message = _user_message.lower()

        with open(file_path, "r") as file:
            for example in file:
                example = example.strip()
                if self.lowercase:
                    example = example.strip()

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
