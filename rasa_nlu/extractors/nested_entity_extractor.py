from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings
import re

from builtins import str
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

from rasa_nlu import utils
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.utils import write_json_to_file

NESTED_ENTITIES_FILE_NAME = "nested_entities.json"


class NestedEntityExtractor(EntityExtractor):
    name = "nested_entity_extractor"
    provides = ["entities"]

    def __init__(self, component_config=None, nested_entities=None):
        # type: (Optional[Dict[Text, Text]]) -> None
        super(NestedEntityExtractor, self).__init__(component_config)

        self.nested_entities = nested_entities if nested_entities else {
            'lookup_tables': [],
            'composite_entities': []
        }

    def train(self, training_data, cfg, **kwargs):
        # type: (TrainingData) -> None
        self.add_lookup_tables(training_data.lookup_tables)
        self.nested_entities['composite_entities'] = training_data.composite_entities

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        entities = message.get("entities", [])[:]
        self.split_nested_entities(entities)
        message.set("entities", entities, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        if self.nested_entities:
            nested_entities_file = os.path.join(model_dir,
                                                NESTED_ENTITIES_FILE_NAME)
            write_json_to_file(nested_entities_file, self.nested_entities,
                               separators=(',', ': '))

        return {"nested_entities_file": NESTED_ENTITIES_FILE_NAME}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[NestedEntitiesMapper]
             **kwargs  # type: **Any
             ):
            # type: (...) -> NestedEntitiesMapper

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("nested_entities_file", NESTED_ENTITIES_FILE_NAME)
        nested_entities_file = os.path.join(model_dir, file_name)

        if os.path.isfile(nested_entities_file):
            nested_entities = utils.read_json_file(nested_entities_file)
        else:
            nested_entities = {
                'lookup_tables': [],
                'composite_entities': []
            }
            warnings.warn("Failed to load nested entities file from '{}'"
                          "".format(nested_entities_file))

        return cls(meta, nested_entities)

    def get_relevance(self,
                      broad_value,
                      nested_composite_examples):
        relevance_score = 0
        for example in nested_composite_examples:
            if example in broad_value:
                relevance_score += 1
        return relevance_score

    def merge_two_dicts(self, x, y):
        z = x.copy()
        z.update(y)
        return z

    def break_on_lookup_tables(self, each_lookup,
                               child_name,
                               broad_value):
        for lookup_entry in each_lookup['elements']:
            find_index = broad_value.find(lookup_entry.lower())
            if(find_index > -1):
                return {
                    child_name: lookup_entry
                }
        return {}

    def split_by_lookup_tables(self, composite_child, broad_value):
        broken_entity = {}
        child_name = composite_child[1:]
        for each_lookup in self.nested_entities['lookup_tables']:
            if(each_lookup['name'] == child_name):
                broken_entity = self.merge_two_dicts(
                    broken_entity,
                    self.break_on_lookup_tables(each_lookup,
                                                child_name,
                                                broad_value))
        return broken_entity

    def split_by_sys(self, composite_child, broad_value):
        broken_entity = {}
        if(composite_child in ['@number', '@year']):
            child_name = composite_child[1:]
            expression = r'\d+'
            if(child_name == 'year'):
                expression = r'\d{4}'
            match = re.findall(expression, broad_value)
            if(match):
                broken_entity[child_name] = match[0]
        return broken_entity

    def split_one_level(self, composite_child, broad_value):
        broken_entity = {}
        broken_entity = self.merge_two_dicts(
            broken_entity,
            self.split_by_lookup_tables(
                composite_child,
                broad_value)
        )
        broken_entity = self.merge_two_dicts(
            broken_entity,
            self.split_by_sys(
                composite_child,
                broad_value)
        )
        return broken_entity

    def get_most_relevant_composite(self,
                                    nested_composites,
                                    broad_value):
        highest_relevance_score = 0
        composite_examples = []
        for nested_composite in nested_composites:
            child_of_nested_composite = filter(
                lambda x: x['name'] == nested_composite,
                self.nested_entities['composite_entities'])
            if(len(child_of_nested_composite) > 0):
                child_synonymns = child_of_nested_composite[0]['composites']
                relevance_score = self.get_relevance(
                    broad_value, child_synonymns)
                if(relevance_score > highest_relevance_score):
                    highest_relevance_score = relevance_score
                    composite_examples = child_synonymns
        return {
            "highest_relevance_score": highest_relevance_score,
            "composite_examples": composite_examples
        }

    def add_most_relevant_composite(self,
                                    most_relevant_composite,
                                    broken_entity,
                                    child_name,
                                    broad_value
                                    ):
        if(most_relevant_composite['highest_relevance_score'] > 0):
            broken_entity[child_name] = {}
            for nested_child in most_relevant_composite['composite_examples']:
                if(nested_child[0] == '@'):
                    broken_entity[child_name] = self.merge_two_dicts(
                        broken_entity[child_name],
                        self.split_one_level(
                            nested_child,
                            broad_value)
                    )
        return broken_entity

    def split_two_levels(self, composite_child, broad_value):
        broken_entity = {}
        child_name = composite_child[1:]
        for each_nested in self.nested_entities['composite_entities']:
            if(each_nested['name'] == child_name):
                nested_composites = each_nested['composites']
                most_relevant_composite = self.get_most_relevant_composite(
                    nested_composites, broad_value)
                broken_entity = self.add_most_relevant_composite(
                    most_relevant_composite,
                    broken_entity,
                    child_name,
                    broad_value
                )
        return broken_entity

    def split_composite_entity(self, composite_entry, entity):
        broken_entity = {}
        broad_value = entity["value"].lower()
        composite_children = composite_entry['composites']
        for composite_child in composite_children:
            if(composite_child[0] == '@'):

                broken_entity = self.merge_two_dicts(
                    broken_entity,
                    self.split_one_level(
                        composite_child,
                        broad_value
                    )
                )

                broken_entity = self.merge_two_dicts(
                    broken_entity,
                    self.split_two_levels(
                        composite_child,
                        broad_value
                    )
                )

        entity["value"] = broken_entity
        self.add_processor_name(entity)

    def split_nested_entities(self, entities):
        for each_entity in entities:
            entity = each_entity["entity"]
            for composite_entry in self.nested_entities['composite_entities']:
                if(composite_entry['name'] == entity):
                    self.split_composite_entity(composite_entry, each_entity)

    def add_lookup_tables(self, lookup_tables):
        """Need to sort by length so that we get the broadest entry first"""
        for lookup in lookup_tables:
            if('elements' in lookup):
                lookup['elements'].sort(key=len, reverse=True)
                self.nested_entities['lookup_tables'].append(lookup)
