from mitie import *
from parsa import Interpreter

class MITIEInterpreter(Interpreter):
    def __init__(self,classifier_file,ner_file,fe_filepath):
        self.extractor = named_entity_extractor(ner_file)
        self.classifier = text_categorizer(classifier_file)
        
    def get_entities(self,tokens):
        d = {}
        entities = self.extractor.extract_entities(tokens)
        for e in entities:
            _range = e[0]
            d[e[1]] = " ".join(tokens[i] for i in _range) 
        return d

    def get_intent(self,tokens):
        label, _ = self.classifier(tokens) # don't use the score
        return label

    def parse(self,text):
        tokens = tokenize(text)
        intent = self.get_intent(tokens)
        entities = self.get_entities(tokens)

        return {'intent':intent,'entities': entities}

