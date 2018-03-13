from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from copy import deepcopy

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.components import Component

if typing.TYPE_CHECKING:
    from rasa_nlu.training_data import Message


class EntityExtractor(Component):
    def add_extractor_name(self, entities):
        # type: (List[Dict[Text, Any]]) -> List[Dict[Text, Any]]
        for entity in entities:
            entity["extractor"] = self.name
        return entities

    def add_processor_name(self, entity):
        # type: (Dict[Text, Any]) -> Dict[Text, Any]
        if "processors" in entity:
            entity["processors"].append(self.name)
        else:
            entity["processors"] = [self.name]
        return entity

    def filter_trainable_entities(self, entity_examples):
        # type: (List[Message]) -> List[Message]
        for message in entity_examples:
            entities = []
            for ent in message.get("entities", []):
                extractor = ent.get("extractor")
                if not extractor or extractor == self.name:
                    entities.append(ent)
            message.set("entities", entities)
        return entity_examples
