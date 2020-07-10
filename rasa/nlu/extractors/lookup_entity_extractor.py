import logging
import os
from typing import Any, Dict, List, Optional, Text

import rasa.utils.common as common_utils
from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import ENTITIES
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message

logger = logging.getLogger(__name__)


class LookupEntityExtractor(EntityExtractor):
    """
    Searches for entities in the user's message from a list of examples.\n
    Required Parameters: \n
    @entities -> list \n
    @files_path -> list
    """

    defaults = {
        "entities": None,
        "files_path": None
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super(LookupEntityExtractor, self).__init__(component_config)

        # check if "entities" and "file_path" are configured in the config.yml
        if component_config is not None and "entities" not in component_config or "files_path" not in component_config:
            self.component_config["entities"] = None
            self.component_config["files_path"] = None
            common_utils.raise_warning(
                "Can't extract Lookup Entities, Please configure the entities and file path for 'LookupEntityExtractor' in the config.yml."
            )

        # check if entities are configured with their respective file path
        # elif component_config["entities"] is not None and component_config["files_path"] is not None:
        elif len(component_config["entities"]) != len(component_config["files_path"]):
            self.component_config["files_path"] = None
            common_utils.raise_warning(
                "Can't extract Lookup Entities, Make sure you have configured the entities and their respective file path properly for 'LookupEntityExtractor' in the config.yml."
            )
        # check if the given file path exists
        elif component_config["files_path"] is not None:
            for i in range(len(component_config["files_path"])):
                file_path = component_config["files_path"][i]
                if os.path.isfile(file_path):
                    pass
                else:
                    self.component_config["files_path"] = None
                    print("File path", file_path)
                    common_utils.raise_warning(
                        f"Can't extract Lookup Entities, File \"{file_path}\" does not exist."
                    )
                    break

    def add_extractor_name(
        self, entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        """
        Adds the Extractor name to the Message class during the prediction.
        """
        for entity in entities:
            entity["extractor"] = self.name
        return entities

    @staticmethod
    def _parse_entities(self, text: Text) -> List[Dict[Text, Any]]:
        """
        pass the user input to  extract the entities

        """

        user_input = text
        entities = self.component_config["entities"]
        files_path = self.component_config["files_path"]

        if files_path is not None and entities is not None:
            if len(entities) != len(files_path):
                return []
            else:
                results = self._parse_all_entities(
                    user_input, entities, files_path)
                return results
        else:
            return []

    @staticmethod
    def _parse_all_entities(user_input: str, entities: list, file_path: list) -> List[Dict[Text, Any]]:
        """
        This method does the actual entity extraction work.\n
        So here I am running the loop over the list of data in the text file\n
        and check whether it exists in the user's message
        """
        results = []
        for i in range(0, len(entities)):
            f = open(file_path[i], "r")
            examples = f.readlines()

            for example in examples:
                if example.lower().strip() in user_input.lower():
                    start_index = user_input.lower().index(example.lower().strip())
                    end_index = start_index + len(example.strip())
                    temp = {}
                    temp["entity"] = entities[i]
                    temp["start"] = start_index
                    temp["end"] = end_index
                    temp["value"] = user_input[start_index:end_index]
                    results.append(temp)
        return results

    def process(self, message, **kwargs):
        """Retrieve the text message, parse the entities."""

        extracted_entities = self._parse_entities(self, message.text)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES,
            message.get(ENTITIES, []) + extracted_entities,
            add_to_output=True,
        )
