from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import typing
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

    def __init__(self, clf=None, regex_dict=None):
        self.clf = clf
        self.regex_dict = {}

    def train(self, training_data):
    # build regex: intent dict from training data
        for example in training_data.regex_features:
            if ("entity" in example):
                self.regex_dict[example["pattern"]] = example["entity"]

    def process(self, text):
        # type: (Doc, Language, List[Dict[Text, Any]]) -> Dict[Text, Any]
        
        entities = self.extract_entities(text)
        return {
             "entities": entities
        }

    def extract_entities(self, text):
        # type: (Text, Language) -> List[Dict[Text, Any]]
        entities = []
        for exp, entity in self.regex_dict.items():
            regexp = re.compile(exp)
            ent = regexp.search(text)
            if ent != None:
                entity = {
                    "entity": str(entity),
                    "value": str(ent.group()),
                    "start": int(ent.start()),
                    "end": int(ent.start() + len(ent.group()))
                }
                entities.append(entity)
        
        return entities