from __future__ import unicode_literals, print_function

import random

import pathlib
from spacy.gold import GoldParse
from spacy.pipeline import EntityRecognizer


class SpacyEntityExtractor(object):
    def __init__(self, nlp=None, extractor_file=None):
        if extractor_file:
            self.ner = EntityRecognizer.load(pathlib.Path(extractor_file), nlp.vocab)
        else:
            self.ner = None

    def convert_examples(self, entity_examples):
        def convert_entity(ent):
            return ent["start"], ent["end"], ent["entity"]

        def convert_example(ex):
            return ex["text"], [convert_entity(ent) for ent in ex["entities"]]

        return [convert_example(ex) for ex in entity_examples]

    def train(self, nlp, entity_examples):
        train_data = self.convert_examples(entity_examples)
        ent_types = [[ent["entity"] for ent in ex["entities"]] for ex in entity_examples]
        entity_types = list(set(sum(ent_types, [])))

        self.ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
        for itn in range(5):
            random.shuffle(train_data)
            for raw_text, entity_offsets in train_data:
                doc = nlp.make_doc(raw_text)
                gold = GoldParse(doc, entities=entity_offsets)
                self.ner.update(doc, gold)
        self.ner.model.end_training()

    def extract_entities(self, nlp, sentence):
        doc = nlp.make_doc(sentence)
        nlp.tagger(doc)
        self.ner(doc)

        entities = [
          {
            "entity": ent.label_,
            "value": ent.text,
            "start": ent.start_char,
            "end": ent.end_char
          }
          for ent in doc.ents]
        return entities
