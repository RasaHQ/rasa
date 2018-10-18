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
            'composite_entries': []
        }

    def train(self, training_data, cfg, **kwargs):
        # type: (TrainingData) -> None
        self.add_lookup_tables(training_data.lookup_tables)
        self.add_composite_entities_from_entities_synonyms(
            training_data.entity_synonyms.items())

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
            nested_entities = None
            warnings.warn("Failed to load nested entities file from '{}'"
                          "".format(nested_entities_file))

        return cls(meta, nested_entities)

    def get_broad_value_nested_composite_examples_relevance(self, broad_value, nested_composite_examples):
        relevance_score = 0
        for example in nested_composite_examples:
            if example in broad_value:
                relevance_score += 1
        return relevance_score

    def merge_two_dicts(self, x, y):
        z = x.copy()   # start with x's keys and values
        z.update(y)    # modifies z with y's keys and values & returns None
        return z

    def split_by_lookup_tables(self, composite_child, broad_value):
        broken_entity = {}
        composite_child_name = composite_child.split(
            ':')[0][1:]  # @model:model => model
        # checking if this composite child can be split by lookup table
        # Checking through our list of lookup tables
        for each_lookup in self.nested_entities['lookup_tables']:
            # if this split composite entity is in the lookup tables
            if(each_lookup['name'] == composite_child_name):
                # kia, honda, toyota
                for lookup_entry in each_lookup['entries']:
                    find_index = broad_value.find(lookup_entry.lower())
                    if(find_index > -1):  # when we find a match of our lookup example in the broad value
                        # split['make'] = 'kia'
                        broken_entity[composite_child_name] = lookup_entry
                        break

        return broken_entity

    def split_by_sys(self, composite_child, broad_value):
        broken_entity = {}
        if(composite_child[0:11] == '@sys.number'):
            composite_child_name = composite_child.split(':')[1]
            expression = r'\d+'
            if(composite_child_name == 'year'):
                expression = r'\d{4}'

            match = re.findall(r'\d+', broad_value)
            if(match):
                broken_entity[composite_child_name] = match[0]
            # TODO ['one', 'two', 'three', 'four', 'five', 'six',
            # 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve']
            # find the examples of one, twi, three, four, etc
        return broken_entity

    def split_one_level(self, composite_child, broad_value):
        broken_entity = {}
        broken_entity = self.merge_two_dicts(
            broken_entity, self.split_by_lookup_tables(composite_child, broad_value))
        broken_entity = self.merge_two_dicts(
            broken_entity, self.split_by_sys(composite_child, broad_value))
        return broken_entity

    def split_composite_entity(self, composite_entry, entity):
        broken_entity = {}
        broad_value = entity["value"].lower()
        composite_children = composite_entry['synonyms']
        # looping thru each child trying to break our broad entity
        for composite_child in composite_children:
            if(composite_child[0] == '@'):
                # @model:model => model
                composite_child_name = composite_child.split(':')[0][1:]

                broken_entity = self.merge_two_dicts(
                    broken_entity, self.split_one_level(composite_child, broad_value))

                # checking if this composite child is also a composite entry
                for each_nested in self.nested_entities['composite_entries']:
                    if(each_nested['value'] == composite_child_name):  # if it is
                        # @doors:doors, @seats:seats, @mpg:mpg
                        nested_composites = each_nested['synonyms']
                        highest_relevance_score = 0
                        correct_nested_composite_examples = []
                        # Try to find the most related nested entity
                        for nested_composite in nested_composites:
                            nested_composite_value = nested_composite.split(':')[
                                0]
                            child_of_nested_composite = filter(
                                lambda x: x['value'] == nested_composite_value,
                                self.nested_entities['composite_entries'])
                            if(len(child_of_nested_composite) == 0):
                                break
                            child_synonymns = child_of_nested_composite[0]['synonyms']
                            relevance_score = self.get_broad_value_nested_composite_examples_relevance(
                                broad_value, child_synonymns)
                            if(relevance_score > highest_relevance_score):
                                highest_relevance_score = relevance_score
                                correct_nested_composite_examples = child_synonymns

                        if(highest_relevance_score > 0):
                            broken_entity[composite_child_name] = {}
                            for nested_composite_child in correct_nested_composite_examples:
                                if(nested_composite_child[0] == '@'):
                                    broken_entity[composite_child_name] = self.merge_two_dicts(
                                        broken_entity[composite_child_name],
                                        self.split_one_level(nested_composite_child, broad_value))

        entity["value"] = {entity["entity"]: broken_entity}
        self.add_processor_name(entity)

    def split_nested_entities(self, entities):
        for each_entity in entities:
            entity = each_entity["entity"]
            for composite_entry in self.nested_entities['composite_entries']:
                if(composite_entry['value'] == entity):
                    self.split_composite_entity(composite_entry, each_entity)

    def add_composite_entities_from_entities_synonyms(self, entity_synonyms):
        for synonym, value in list(entity_synonyms):
            if(value[0] == '@' and synonym.startswith(value)):
                # unpacking the smuggled entities here
                synonym = synonym.replace(value + "_", "")
                value = value[1:]

                element_present = False
                for composite_index, element in enumerate(self.nested_entities['composite_entries']):
                    if element['value'] == value:
                        self.nested_entities['composite_entries'][composite_index]['synonyms'].append(
                            synonym)
                        element_present = True
                if not element_present:
                    self.nested_entities['composite_entries'].append({
                        'synonyms': [
                            synonym
                        ],
                        'value': value
                    })

    def add_lookup_tables(self, lookup_tables):
        for lookup in lookup_tables:
            # need to sort by length so that we get the broadest entry first
            lookup['entries'].sort(key=len, reverse=True)
            self.nested_entities['lookup_tables'].append(lookup)
