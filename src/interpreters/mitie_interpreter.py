from mitie import *
from rasa_nlu import Interpreter
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
import re


class MITIEInterpreter(Interpreter):
    def __init__(self, intent_classifier=None, entity_extractor=None, feature_extractor=None, **kwargs):
        self.extractor = named_entity_extractor(entity_extractor, feature_extractor)
        self.classifier = text_categorizer(intent_classifier, feature_extractor)
        self.tokenizer = MITIETokenizer()

    def get_entities(self, text):
        tokens = self.tokenizer.tokenize(text)
        ents = []
        entities = self.extractor.extract_entities(tokens)
        for e in entities:
            _range = e[0]
            _regex = u"\s*".join(tokens[i] for i in _range)
            expr = re.compile(_regex)
            m = expr.search(text)
            start, end = m.start(), m.end()
            ents.append({
                "entity": e[1],
                "value": text[start:end],
                "start": start,
                "end": end
            })

        return ents

    def get_intent(self, text):
        tokens = tokenize(text)
        label, _ = self.classifier(tokens)  # don't use the score
        return label

    def parse(self, text):
        intent = self.get_intent(text)
        entities = self.get_entities(text)

        return {'text': text, 'intent': intent, 'entities': entities}
