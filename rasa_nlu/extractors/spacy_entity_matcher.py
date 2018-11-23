from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

import os
import spacy
import pickle
from spacy.matcher import Matcher
from rasa_nlu.extractors import EntityExtractor

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc

PATTERN_NER_FILE = 'pattern_ner.pkl'


class SpacyEntityMatcher(EntityExtractor):
    name = "pattern_ner_spacy"

    provides = ["entities"]

    requires = ["tokens"]

    def __init__(self, component_config=None, matcher=None):
        super(SpacyPatternNER, self).__init__(component_config)
        if matcher:
            self.matcher = matcher
            self.spacy_nlp = spacy.blank('en')
            self. spacy_nlp.vocab = self.matcher.vocab
        else:
            self.spacy_nlp = spacy.blank('en')
            self.matcher = Matcher(self.spacy_nlp.vocab)

    def train(self, training_data, cfg, **kwargs):
        for lookup_table in training_data.lookup_tables:
            key = lookup_table['name']
            pattern = []
            for element in lookup_table['elements']:
                tokens = [{'LOWER': token.lower()}
                          for token in str(element).split()]
                pattern.append(tokens)
            self.matcher.add(key, None, *pattern)

    def process(self, message, **kwargs):
        doc = self.spacy_nlp(message.text)
        matches = self.matcher(doc)
        entities = []
        for ent_id, start, end in matches:
            entities.append({
                'start': start,
                'end': end,
                'value': doc[start:end].text,
                'entity': self.matcher.vocab.strings[ent_id],
                'confidence': None,
                'extractor': self.name
            })
        message.set("entities", message.get("entities", []) + entities,
                    add_to_output=True)

    def persist(self, model_dir):
        if self.matcher:
            modelFile = os.path.join(model_dir, PATTERN_NER_FILE)
            self.saveModel(modelFile)
        return {"pattern_ner_file": PATTERN_NER_FILE}

    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs
             ):
        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("pattern_ner_file", PATTERN_NER_FILE)
        modelFile = os.path.join(model_dir, file_name)
        if os.path.exists(modelFile):
            modelLoad = open(modelFile, "rb")
            matcher = pickle.load(modelLoad)
            return cls(meta, matcher)
        else:
            return cls(meta)

    def saveModel(self, modelFile):
        modelSave = open(modelFile, "wb")
        pickle.dump(self.matcher, modelSave)
        modelSave.close()
