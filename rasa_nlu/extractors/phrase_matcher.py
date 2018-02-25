from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import defaultdict

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message
from rasa_nlu import utils


class PhraseMatcher(EntityExtractor):
    name = "ner_phrase_matcher"

    provides = ["entities"]

    requires = []

    def __init__(self, entity_phrases=None, ignore_case=True):
        import pygtrie
        self.phrase_trie = pygtrie.CharTrie()
        self.ignore_case = ignore_case
        if entity_phrases:
            self._build_trie(entity_phrases)

    def _build_trie(self, entity_phrases):
        for entity, phrases in entity_phrases.items():
            for phrase in phrases:
                if self.ignore_case:
                    phrase = phrase.lower()
                self.phrase_trie[phrase] = entity

    def train(self, training_data, config, **kwargs):
        ignore_case = config.get("phrase_matcher", {}).get("ignore_case")
        if ignore_case is not None:
            self.ignore_case = ignore_case
        self._build_trie(training_data.entity_phrases)

    def process(self, message, **kwargs):
        # type: (Message) -> None

        extracted = []
        text = message.text
        if self.ignore_case:
            text = text.lower()

        for i in range(len(text)):
            match = self.phrase_trie.longest_prefix(text[i:])
            if match:
                start, end = i, i + len(match[0])
                value = message.text[start:end] if self.ignore_case else match[0]
                entity_type = match[1]
                extracted.append(utils.build_entity(start, end, value, entity_type))

        self.append_entities(message, extracted)

    def persist(self, model_dir):
        entity_phrases = defaultdict(list)
        for phrase, entity in self.phrase_trie.items():
            entity_phrases[entity].append(phrase)

        entity_phrases_file = "entity_phrases.json"
        entity_phrases_file_path = os.path.join(model_dir, entity_phrases_file)
        utils.write_json_to_file(entity_phrases_file_path, entity_phrases)

        return {
            "entity_phrases": entity_phrases_file,
            "phrase_matcher": {
                "ignore_case": self.ignore_case
            }
        }

    @classmethod
    def load(cls, model_dir, model_metadata, cached_component, **kwargs):
        if not model_metadata.get("entity_phrases"):
            raise ValueError("Entity phrases not defined in metadata, but component present in pipeline")
        kwargs = {}
        ignore_case = model_metadata.get("phrase_matcher", {}).get("ignore_case")
        if ignore_case is not None:
            kwargs["ignore_case"] = ignore_case

        entity_phrases_file = os.path.join(model_dir, model_metadata.get("entity_phrases"))
        entity_phrases = utils.read_json_file(entity_phrases_file)

        return cls(entity_phrases, **kwargs)
