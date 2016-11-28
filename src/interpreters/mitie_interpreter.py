from mitie import *
from rasa_nlu import Interpreter


class MITIEInterpreter(Interpreter):
    def __init__(self, intent_classifier=None, entity_extractor=None, feature_extractor=None, **kwargs):
        self.extractor = named_entity_extractor(entity_extractor, feature_extractor)
        self.classifier = text_categorizer(intent_classifier, feature_extractor)

    def get_entities(self, tokens):
        d = {}
        entities = self.extractor.extract_entities(tokens)
        for e in entities:
            _range = e[0]
            d[e[1]] = " ".join(tokens[i] for i in _range)
        return d

    def get_intent(self, tokens):
        label, _ = self.classifier(tokens)  # don't use the score
        return label

    def parse(self, text):
        tokens = tokenize(text)
        intent = self.get_intent(tokens)
        entities = self.get_entities(tokens)

        return {'text': text, 'intent': intent, 'entities': entities}
