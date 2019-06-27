from typing import Any, Dict, List, Text

from rasa.nlu.components import Component
from rasa.nlu.training_data import Message


class EntityExtractor(Component):
    def add_extractor_name(
        self, entities: List[Dict[Text, Any]]
    ) -> List[Dict[Text, Any]]:
        for entity in entities:
            entity["extractor"] = self.name
        return entities

    def add_processor_name(self, entity: Dict[Text, Any]) -> Dict[Text, Any]:
        if "processors" in entity:
            entity["processors"].append(self.name)
        else:
            entity["processors"] = [self.name]

        return entity

    @staticmethod
    def filter_irrelevant_entities(extracted, requested_dimensions):
        """Only return dimensions the user configured"""

        if requested_dimensions:
            return [
                entity
                for entity in extracted
                if entity["entity"] in requested_dimensions
            ]
        else:
            return extracted

    @staticmethod
    def find_entity(ent, text, tokens):
        offsets = [token.offset for token in tokens]
        ends = [token.end for token in tokens]

        if ent["start"] not in offsets:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity start.".format(ent, text)
            )
            raise ValueError(message)

        if ent["end"] not in ends:
            message = (
                "Invalid entity {} in example '{}': "
                "entities must span whole tokens. "
                "Wrong entity end.".format(ent, text)
            )
            raise ValueError(message)

        start = offsets.index(ent["start"])
        end = ends.index(ent["end"]) + 1
        return start, end

    def filter_trainable_entities(
        self, entity_examples: List[Message]
    ) -> List[Message]:
        """Filters out untrainable entity annotations.

        Creates a copy of entity_examples in which entities that have
        `extractor` set to something other than
        self.name (e.g. 'CRFEntityExtractor') are removed.
        """

        filtered = []
        for message in entity_examples:
            entities = []
            for ent in message.get("entities", []):
                extractor = ent.get("extractor")
                if not extractor or extractor == self.name:
                    entities.append(ent)
            data = message.data.copy()
            data["entities"] = entities
            filtered.append(
                Message(
                    text=message.text,
                    data=data,
                    output_properties=message.output_properties,
                    time=message.time,
                )
            )

        return filtered


    @staticmethod
    def add_roles_to_entities(role_message: Message, message: Message) -> Message:
       """mark all predicted roles as roles not entities"""
       starts = {ent["start"]: ent for ent in role_message.get("entities", [])}

       entities = []
       for ent in message.get("entities", []):
           start_idx = ent["start"]
           if start_idx in starts:
               ent_with_role = ent.copy()
               ent_with_role["role"] = starts[start_idx]["entity"]
               entities.append(ent_with_role)
           else:
               entities.append(ent)

       data = message.data.copy()
       data["entities"] = entities
       return Message(
                  text=message.text,
                  data=data,
                  output_properties=message.output_properties,
                  time=message.time,
              )



    @staticmethod
    def replace_entities_with_roles(message: Message) -> Message:
       """replace all entities which have a role with a role"""

       entities = []
       text = message.text
       for ent in message.get("entities", []):
           if ent.get("role"):
               role_ent = ent.copy()
               role_ent["entity"] = ent["role"]
               role_ent["value"] = ent["entity"]
               # TODO update start and end values
               text = message.text[:ent["start"]] + \
                      ent["entity"] + \
                      message.text[ent["end"]:]
               entities.append(role_ent)

       data = message.data.copy()
       data["entities"] = entities
       return Message(
                  text=text,
                  data=data,
                  output_properties=message.output_properties,
                  time=message.time,
              )

    @staticmethod
    def create_role_examples(
        entity_examples: List[Message]
    ) -> List[Message]:
        """Creates role examples.
        """

        return [ EntityExtractor.replace_entities_with_roles(message)
                 for message in entity_examples ]
