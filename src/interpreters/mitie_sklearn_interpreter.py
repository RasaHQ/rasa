from mitie import named_entity_extractor, text_categorizer
from rasa_nlu import Interpreter
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
import json
import codecs

class MITIESklearnInterpreter(Interpreter):
    def __init__(self, metadata, entity_synonyms=None):
        self.extractor = named_entity_extractor(metadata["entity_extractor"])  # ,metadata["feature_extractor"])
        self.classifier = text_categorizer(metadata["intent_classifier"])  # ,metadata["feature_extractor"])
        self.tokenizer = MITIETokenizer()
        if entity_synonyms:
            self.entity_synonyms = json.loads(codecs.open(entity_synonyms, encoding='utf-8').read())

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
        tokens = self.tokenizer.tokenize(text)
        intent = self.get_intent(tokens)
        entities = self.get_entities(tokens)
        for i in range(len(entities)):
            entity_value = entities[i]["value"]
            if entity_value in self.entity_synonyms:
                entities[i]["value"] = self.entity_synonyms[entity_value]

        return {'text': text, 'intent': intent, 'entities': entities}
