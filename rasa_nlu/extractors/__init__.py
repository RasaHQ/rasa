from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.components import Component


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
