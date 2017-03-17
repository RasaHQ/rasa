import codecs
import json
import os
import warnings

from rasa_nlu.components import Component
from rasa_nlu.training_data import TrainingData


class EntitySynonymMapper(Component):
    name = "ner_synonyms"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    def __init__(self, synonyms=None):
        self.synonyms = synonyms if synonyms else {}

    def train(self, training_data):
        # type: (TrainingData) -> None

        for key, value in training_data.entity_synonyms.items():
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example["entities"]:
                entity_val = example["text"][entity["start"]:entity["end"]]
                self.add_entities_if_synonyms(entity_val, entity.get("value"))

    def process(self, entities):
        # type: (dict) -> dict

        updated_entities = entities[:]
        self.replace_synonyms(updated_entities)

        return {
            "entities": updated_entities
        }

    def persist(self, model_dir):
        # type: (str) -> dict

        if self.synonyms:
            entity_synonyms_file = os.path.join(model_dir, "index.json")
            with open(entity_synonyms_file, 'w') as f:
                json.dump(self.synonyms, f)
            return {"entity_synonyms": "index.json"}
        else:
            return {"entity_synonyms": None}

    @classmethod
    def load(cls, model_dir, entity_synonyms):
        # type: (str, str) -> EntitySynonymMapper

        if model_dir and entity_synonyms:
            entity_synonyms_file = os.path.join(model_dir, "index.json")
            if os.path.isfile(entity_synonyms_file):
                with codecs.open(entity_synonyms_file, encoding='utf-8') as f:
                    synonyms = json.loads(f.read())
                return EntitySynonymMapper(synonyms)
            else:
                warnings.warn("Failed to load synonyms file from '{}'".format(entity_synonyms_file))
        return EntitySynonymMapper()

    def replace_synonyms(self, entities):
        for i in range(len(entities)):
            entity_value = entities[i]["value"]
            if entity_value.lower() in self.synonyms:
                entities[i]["value"] = self.synonyms[entity_value.lower()]

    def add_entities_if_synonyms(self, entity_a, entity_b):
        if entity_b is not None:
            original = entity_a.lower() if type(entity_a) == unicode else unicode(entity_a)
            replacement = entity_b.lower() if type(entity_b) == unicode else unicode(entity_b)

            if original != replacement:
                self.synonyms[original] = replacement
