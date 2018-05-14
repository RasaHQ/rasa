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

    requires = ["spacy_doc", "tokens"]

    defaults = {
        "ignore_case": True,
        "use_tokens": True
    }

    def __init__(self, component_config=None, entity_phrases=None):
        super(PhraseMatcher, self).__init__(component_config)

        self.phrase_trie = self._trie_factory(self.component_config["use_tokens"])
        if entity_phrases:
            self._build_trie(entity_phrases)

    @staticmethod
    def _trie_factory(use_tokens):
        """Return appropriate Trie for operation mode."""
        import pygtrie
        if use_tokens:
            return pygtrie.Trie()
        else:
            return pygtrie.CharTrie()

    def _adjust_phrases(self, phrases):
        """Adjusts phrases for case sensitivity and token-based matching."""
        fitted = []
        for phrase in phrases:
            if self.component_config["ignore_case"]:
                phrase = phrase.lower()
            if self.component_config["use_tokens"]:
                phrase = phrase.split(" ")
            fitted.append(phrase)
        return fitted

    def _build_trie(self, entity_phrases):
        for entity, phrases in entity_phrases.items():
            phrases = self._adjust_phrases(phrases)
            for phrase in phrases:
                self.phrase_trie[phrase] = entity

    def train(self, training_data, config, **kwargs):
        self.component_config = config.for_component(self.name, self.defaults)
        self._build_trie(training_data.entity_phrases)

    def process(self, message, **kwargs):
        # type: (Message) -> None
        if self.component_config["use_tokens"]:
            self._process_tokens(message)
        else:
            self._process_chars(message)

    def _get_tokens(self, message):
        spacy_doc = message.get("spacy_doc")
        tokens = message.get("tokens")
        if spacy_doc is not None:
            return [(t.text, t.idx) for t in spacy_doc]
        elif tokens is not None:
            return [(t.text, t.offset) for t in tokens]
        else:
            raise ValueError("Neither 'spacy_doc' nor 'tokens' present for phrase matcher in tokenized mode.")

    def _process_tokens(self, message):
        extracted = []
        tokens = self._get_tokens(message)
        tokens, indices = zip(*tokens)
        if self.component_config["ignore_case"]:
            tokens = [t.lower() for t in tokens]
        for i in range(len(tokens)):
            match = self.phrase_trie.longest_prefix(tokens[i:])
            if match:
                start = indices[i]
                end_i = i + len(match[0]) - 1
                end = indices[end_i] + len(tokens[end_i])
                value = message.text[start:end]
                entity_type = match[1]
                extracted.append(utils.build_entity(start, end, value, entity_type))

        self.append_entities(message, extracted)

    def _process_chars(self, message):
        extracted = []
        text = message.text
        if self.component_config["ignore_case"]:
            text = text.lower()

        for i in range(len(text)):
            match = self.phrase_trie.longest_prefix(text[i:])
            if match:
                start, end = i, i + len(match[0])
                value = message.text[start:end]
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
        }

    @classmethod
    def load(cls, model_dir, model_metadata, cached_component, **kwargs):
        if not model_metadata.get("entity_phrases"):
            raise ValueError("Entity phrases not defined in metadata, but component present in pipeline")

        component_config = model_metadata.for_component(cls.name)

        entity_phrases_file = os.path.join(model_dir, model_metadata.get("entity_phrases"))
        entity_phrases = utils.read_json_file(entity_phrases_file)

        return cls(component_config, entity_phrases)
