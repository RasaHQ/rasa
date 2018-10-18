from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings

from builtins import str
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

from rasa_nlu import utils
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.utils import write_json_to_file

ENTITY_SYNONYMS_FILE_NAME = "entity_synonyms.json"


class EntitySynonymMapper(EntityExtractor):
    name = "ner_synonyms"

    provides = ["entities"]

    def __init__(self, component_config=None, synonyms=None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(EntitySynonymMapper, self).__init__(component_config)

        self.synonyms = synonyms if synonyms else {}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get("entities", []):
                entity_val = example.text[entity["start"]:entity["end"]]
                self.add_entities_if_synonyms(entity_val,
                                              str(entity.get("value")))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated_entities = message.get("entities", [])[:]
        self.replace_synonyms(updated_entities)
        message.set("entities", updated_entities, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]

        if self.synonyms:
            entity_synonyms_file = os.path.join(model_dir,
                                                ENTITY_SYNONYMS_FILE_NAME)
            write_json_to_file(entity_synonyms_file, self.synonyms,
                               separators=(',', ': '))
            return {"synonyms_file": ENTITY_SYNONYMS_FILE_NAME}
        else:
            return {"synonyms_file": None}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[EntitySynonymMapper]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EntitySynonymMapper

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("synonyms_file")
        if not file_name:
            synonyms = None
            return cls(meta, synonyms)

        entity_synonyms_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entity_synonyms_file):
            synonyms = utils.read_json_file(entity_synonyms_file)
        else:
            synonyms = None
            warnings.warn("Failed to load synonyms file from '{}'"
                          "".format(entity_synonyms_file))
        return cls(meta, synonyms)

    def replace_synonyms(self, entities):
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])
            if entity_value.lower() in self.synonyms:
                entity["value"] = self.synonyms[entity_value.lower()]
                self.add_processor_name(entity)

    def add_entities_if_synonyms(self, entity_a, entity_b):
        if entity_b is not None:
            original = utils.as_text_type(entity_a)
            replacement = utils.as_text_type(entity_b)

            if original != replacement:
                original = original.lower()
                if (original in self.synonyms
                        and self.synonyms[original] != replacement):
                    warnings.warn("Found conflicting synonym definitions "
                                  "for {}. Overwriting target {} with {}. "
                                  "Check your training data and remove "
                                  "conflicting synonym definitions to "
                                  "prevent this from happening."
                                  "".format(repr(original),
                                            repr(self.synonyms[original]),
                                            repr(replacement)))

                self.synonyms[original] = replacement
