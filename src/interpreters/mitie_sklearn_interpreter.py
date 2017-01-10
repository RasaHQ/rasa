from mitie import named_entity_extractor, text_categorizer
from rasa_nlu import Interpreter
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


class MITIESklearnInterpreter(Interpreter):
    def __init__(self, metadata):
        self.extractor = named_entity_extractor(metadata["entity_extractor"])  # ,metadata["feature_extractor"])
        self.classifier = text_categorizer(metadata["intent_classifier"])  # ,metadata["feature_extractor"])
        self.tokenizer = MITIETokenizer()

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

        return {'text': text, 'intent': intent, 'entities': entities}
