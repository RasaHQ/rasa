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
        "lookup": None
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(LookupEntityExtractor, self).__init__(component_config)

        if component_config is not None and "lookup" in component_config:
            if component_config["lookup"] is not None:
                # check if the entities and respective file path exists
                for key, value in list(component_config["lookup"].items()):
                    self.add_entities_if_lookup_entity(key, value)
            else:
                self.component_config["lookup"] = None

            if not bool(component_config["lookup"]):
                # if the lookup dictionary is empty after filtering
                # the lookup entities, assign it None value
                self.component_config["lookup"] = None
        else:
            common_utils.raise_warning(
                "Can't extract Lookup Entities,\
                Please configure the lookup entities in the config.yml."
            )

    def _validate_lookup_entry(self, entity: Text, file_path: Text) -> None:
        if file_path is not None:
            if os.path.isfile(file_path):
                pass
            else:
                # remove the entity from the lookup dictionary,
                # if the file path doesn't exist
                self.component_config["lookup"].pop(entity)
                common_utils.raise_warning(
                    f"can't extract lookup entity: '{entity}',\
                    make sure the provided file '{file_path}' exists.")
        else:
            # remove the entity from the lookup dictionary,
            # if the file path is not provided
            self.component_config["lookup"].pop(entity)
            common_utils.raise_warning(
                f"can't extract '{entity}' entity,\
                please provide the example file.")

    def _parse_entities(self, user_input: Text) -> List[Dict[Text, Any]]:
        """Extract entities from the user input."""
        if self.component_config["lookup"] is not None:
            for entity, file_path in list(
                    self.component_config["lookup"].items()):
                results = self._parse_all_entities(
                    user_input, entity, file_path)
                return results
        else:
            return []

    @staticmethod
    def _parse_all_entities(
        user_input: str, entity: list, file_path: list
    ) -> List[Dict[Text, Any]]:
        """
        This method does the actual entity extraction work.
        So here I am running the loop over the list of data in the text file
        and check whether it exists in the user's message
        """
        results = []
        with open(file_path, "r") as f:
            examples = f.readlines()

            for example in examples:
                example = example.lower().strip()
                if example in user_input.lower():
                    start_index = user_input.lower().index(example)
                    end_index = start_index + len(example.strip())
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
