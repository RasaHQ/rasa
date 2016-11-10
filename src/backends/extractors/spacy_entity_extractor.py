from __future__ import unicode_literals, print_function
import json
import pathlib
import random

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger

class SpacyEntityExtractor(self,nlp):
    def __init__(self,nlp):
        self.nlp = nlp
        self.ner = None

    def train(self,entity_examples):
        train_data = [
           ( ex["text"],[(ent["start"],ent["end"],ent["entity"]) for ent in ex["entities"]])
           for ex in entity_examples
         ]
        entity_types = list(set([ent["entity"] for ent in ex for ex in entity_examples]))

        self.ner = EntityRecognizer(self.nlp.vocab, entity_types=entity_types)
        for itn in range(5):
            random.shuffle(train_data)
            for raw_text, entity_offsets in train_data:
                doc = self.nlp.make_doc(raw_text)
                gold = GoldParse(doc, entities=entity_offsets)
                self.ner.update(doc, gold)
        self.ner.model.end_training()
        
    def extract_entities(self,sentence):
        doc = self.nlp.make_doc(sentence)
        self.nlp.tagger(doc)
        self.ner(doc)
        for word in doc:
            print(word.text, word.tag_, word.ent_type_, word.ent_iob)

         
         
         
         
         

