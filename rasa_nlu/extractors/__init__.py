from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
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
