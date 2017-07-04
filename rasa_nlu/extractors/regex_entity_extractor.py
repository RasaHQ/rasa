from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import io
import json
import os
import re

import typing
from builtins import str
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import TrainingData, Message

if typing.TYPE_CHECKING:
    from spacy.language import Language


class RegExEntityExtractor(EntityExtractor):
    name = "ner_regex"

    provides = ["entities"]

    def __init__(self, regex_dict=None):
        self.regex_dict = regex_dict if regex_dict else {}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """ Build regex: intent dictionary from training data"""

        for example in training_data.regex_features:
            if "entity" in example:
                self.regex_dict[example["pattern"]] = example["entity"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message.text))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, text):
        # type: (Text, Language) -> List[Dict[Text, Any]]

        entities = []
        for exp, _entity in self.regex_dict.items():
            regexp = re.compile(exp)
            ent = regexp.finditer(text)
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

        file_name = self.name + ".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(str(json.dumps({"regex_dictionary": self.regex_dict})))
        return {"ner_regex_persisted": file_name}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text) -> RegExEntityExtractor

        if model_dir and model_metadata.get("ner_regex_persisted"):
            persisted = os.path.join(model_dir, model_metadata.get("ner_regex_persisted"))
            if os.path.isfile(persisted):
                with io.open(persisted, encoding='utf-8') as f:
                    persisted_data = json.loads(f.read())
                    return RegExEntityExtractor(persisted_data["regex_dictionary"])
        else:
            return RegExEntityExtractor()
