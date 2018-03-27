from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.components import Component
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

    @staticmethod
    def find_entity(ent, text, tokens):
        offsets = [token.offset for token in tokens]
        ends = [token.end for token in tokens]

        if ent["start"] not in offsets:
            message = ("Invalid entity {} in example '{}': "
                       "entities must span whole tokens. "
                       "Wrong entity start.".format(ent, text))
            raise ValueError(message)

        if ent["end"] not in ends:
            message = ("Invalid entity {} in example '{}': "
                       "entities must span whole tokens. "
                       "Wrong entity end.".format(ent, text))
            raise ValueError(message)

        start = offsets.index(ent["start"])
        end = ends.index(ent["end"]) + 1
        return start, end

    def filter_trainable_entities(self, entity_examples):
        # type: (List[Message]) -> List[Message]
        """Filters out untrainable entity annotations.

        Creates a copy of entity_examples in which entities that have
        `extractor` set to something other than self.name (e.g. 'ner_crf')
        are removed."""

        filtered = []
        for message in entity_examples:
            entities = []
            for ent in message.get("entities", []):
                extractor = ent.get("extractor")
                if not extractor or extractor == self.name:
                    entities.append(ent)
            data = message.data.copy()
            data['entities'] = entities
            filtered.append(
                Message(text=message.text,
                        data=data,
                        output_properties=message.output_properties,
                        time=message.time))

        return filtered
