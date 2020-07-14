import os
import logging
from typing import Any, Dict, List, Optional, Text

import rasa.utils.common as common_utils
from rasa.nlu.constants import ENTITIES
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
        # it contains the dictonary of lookup entity names
        # and their respective data files
        # example:
        # - name: LookupEntityExtractor
        #   lookup:
        #      city: /some/path/city.txt
        #      person: /some/other/path/person.txt
        "lookup": None,
        # lower case the entity value from the lookup file and
        # user query while comparing them
        "lowercase": True
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(LookupEntityExtractor, self).__init__(component_config)

        if component_config is not None and "lookup" in component_config:
            if component_config["lookup"] is not None:
                # check if the entities and respective file path exists
                for key, value in list(component_config["lookup"].items()):
                    self._validate_lookup_entry(key, value)
            else:
                message = (
                    "Can't extract Lookup Entities, "
                    "Please provide a valid entries for "
                    "entities and their respective file paths."
                )
                raise ValueError(message)
        else:
            message = (
                "Can't extract Lookup Entities, "
                "Please configure the lookup entities in the config.yml."
            )
            raise ValueError(message)

    def _validate_lookup_entry(self, entity: Text, file_path: Text) -> None:
        if file_path is not None:
            if not os.path.isfile(file_path):
                # remove the entity from the lookup dictionary,
                # if the file path doesn't exist
                self.component_config["lookup"].pop(entity)
                message = (
                    f"The file path '{file_path}' for entity '{entity}' "
                    "does not exist. "
                    "Please provide a valid file path."
                )
                common_utils.raise_warning(message)
        else:
            # remove the entity from the lookup dictionary,
            # if the file path is not provided
            self.component_config["lookup"].pop(entity)
            message = (
                f"No file path for entity '{entity}' was given. "
                "Please provide a valid file path."
            )
            common_utils.raise_warning(message)

    def _parse_entities(self, user_input: Text) -> List[Dict[Text, Any]]:
        """Extract entities from the user input."""
        for entity, file_path in self.component_config["lookup"].items():
            results = self._extract_entities(
                self, user_input, entity, file_path)
            return results

    @staticmethod
    def _extract_entities(
        self, user_input: Text,
        entity: Text, file_path: Text
    ) -> List[Dict[Text, Any]]:
        """
        This method does the actual entity extraction work.
        So here we are running the loop over the list of data in the text file
        and check whether it exists in the user's message
        """
        results = []
        user_input_temp = user_input
        if self.component_config["lowercase"]:
            # check for lower case
            user_input_temp = user_input.lower()

        with open(file_path, "r") as file:
            for example in file:
                # check for lower case
                if self.component_config["lowercase"]:
                    example = example.lower().strip()
                else:
                    example = example.strip()
                if example in user_input_temp:
                    start_index = user_input_temp.index(example)
                    end_index = start_index + len(example)
                    results.append({
                        "entity": entity,
                        "start": start_index,
                        "end": end_index,
                        "value": user_input[start_index:end_index]
                    })
        return results

    def process(self, message: Message, **kwargs: Any) -> None:
        """Retrieve the text message, parse the entities."""

        extracted_entities = self._parse_entities(message.text)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES,
            message.get(ENTITIES, []) + extracted_entities,
            add_to_output=True,
        )
