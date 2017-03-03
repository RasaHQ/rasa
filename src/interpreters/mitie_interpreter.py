from mitie import *

from rasa_nlu import Interpreter
from rasa_nlu.interpreters.mitie_interpreter_utils import get_entities
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


class MITIEInterpreter(Interpreter):
    def __init__(self,
                 intent_classifier=None,
                 entity_extractor=None,
                 feature_extractor=None,
                 entity_synonyms=None,
                 **kwargs):
        self.extractor = None
        self.classifier = None
        if entity_extractor:
            self.extractor = named_entity_extractor(entity_extractor, feature_extractor)
        if intent_classifier:
            self.classifier = text_categorizer(intent_classifier, feature_extractor)
        self.tokenizer = MITIETokenizer()
        self.ent_synonyms = None
        if entity_synonyms:
            Interpreter.load_synonyms(entity_synonyms)

    def get_intent(self, tokens):
        if self.classifier:
            label, score = self.classifier(tokens)
        else:
            label, score = "None", 0.0
        return label, score

    def parse(self, text, **kwargs):
        tokens = self.tokenizer.tokenize(text)
        intent, score = self.get_intent(tokens)
        entities = get_entities(text, tokens, self.extractor)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': score}
