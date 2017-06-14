from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import typing
import os
import io
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Text
import re

import rasa_nlu.converters as converters
from rasa_nlu.training_data import TrainingData

from rasa_nlu.extractors import EntityExtractor

if typing.TYPE_CHECKING:
    from spacy.language import Language


class RegExEntityExtractor(EntityExtractor):
    name = "ner_regex"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    def __init__(self, regex_dict={}):
        self.regex_dict = regex_dict

    def train(self, training_data):
        """ 
        Build regex: intent dictionary from training data
        """
        for example in training_data.regex_features:
            if "entity" in example:
                self.regex_dict[example["pattern"]] = example["entity"]

    def process(self, text, entities):
        # type: (Doc, Language, List[Dict[Text, Any]]) -> Dict[Text, Any]
        extracted = self.add_extractor_name(self.extract_entities(text))
        entities.extend(extracted)
        return {
            "entities": entities
        }

    def extract_entities(self, text):
        # type: (Text, Language) -> List[Dict[Text, Any]]
        entities = []
        for exp, _entity in self.regex_dict.items():
            regexp = re.compile(exp)
            ent = regexp.finditer(text)
            if ent != []:
                for match in ent:
                    entity = {
                        "entity": _entity,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.start() + len(match.group())
                    }
                    entities.append(entity)
        return entities

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        file_name = self.name+".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(unicode(json.dumps({"regex_dictionary": self.regex_dict})))
        return {"ner_regex_persisted": file_name}

    @classmethod
    def load(cls, model_dir, ner_regex_persisted):
        # type: (Text) -> RegExEntityExtractor
        persisted = os.path.join(model_dir, ner_regex_persisted)
        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = json.loads(f.read())
                return RegExEntityExtractor(persisted_data["regex_dictionary"])

