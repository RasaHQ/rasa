from __future__ import unicode_literals, print_function
import json
import pathlib
import random
from pprint import pprint
from os import path

import spacy
from spacy.pipeline import EntityRecognizer
from spacy.gold import GoldParse
from spacy.tagger import Tagger

class SpacyEntityExtractor(object):
    def __init__(self,nlp=None,extractor_file=None):
        if (extractor_file):
            self.ner = EntityRecognizer.load(pathlib.Path(extractor_file),nlp.vocab)
        else:            
            self.ner = None

    def train(self,nlp,entity_examples):

        ent_types = [[ent["entity"] for ent in ex["entities"]] for ex in entity_examples]
        entity_types = list(set(sum(ent_types,[])))
        train_data = [
           ( 
             ex["text"],
             [(ent["start"],ent["end"],ent["entity"]) for ent in ex["entities"]]
           )     
           for ex in entity_examples
        ]

        self.ner = EntityRecognizer(nlp.vocab, entity_types=entity_types)
        for itn in range(5):
            random.shuffle(train_data)
            for raw_text, entity_offsets in train_data:
                doc = nlp.make_doc(raw_text)
                gold = GoldParse(doc, entities=entity_offsets)
                self.ner.update(doc, gold)
        self.ner.model.end_training()
        
    def extract_entities(self,nlp,sentence):    
        doc = nlp.make_doc(sentence)
        nlp.tagger(doc)
        self.ner(doc)
        entities = { ent.label_: ent.text for ent in doc.ents }
        return entities
        
         
         
         
         
         

