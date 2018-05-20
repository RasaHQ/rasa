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


def biluo_tags_from_offsets(tokens, entities, missing='O'):
    """Implementation adapted from spacy.
    See https://github.com/explosion/spaCy/blob/master/spacy/gold.pyx#L493
    """

    starts = {token.offset: i for i, token in enumerate(tokens)}
    ends = {token.offset+len(token.text): i for i, token in enumerate(tokens)}
    biluo = ['-' for _ in tokens]
    # Handle entity cases
    for start_char, end_char, label in entities:
        start_token = starts.get(start_char)
        end_token = ends.get(end_char)
        # Only interested if the tokenization is correct
        if start_token is not None and end_token is not None:
            if start_token == end_token:
                biluo[start_token] = 'U-%s' % label
            else:
                biluo[start_token] = 'B-%s' % label
                for i in range(start_token+1, end_token):
                    biluo[i] = 'I-%s' % label
                biluo[end_token] = 'L-%s' % label
    # Now distinguish the O cases from ones where we miss the tokenization
    entity_chars = set()
    for start_char, end_char, label in entities:
        for i in range(start_char, end_char):
            entity_chars.add(i)
    for n, token in enumerate(tokens):
        for i in range(token.offset, token.offset + len(token.text)):
            if i in entity_chars:
                break
        else:
            biluo[n] = missing

    return biluo
