from mitie import named_entity_extractor
import cloudpickle
from rasa_nlu import Interpreter
from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
import json
import codecs


class MITIESklearnInterpreter(Interpreter):
    def __init__(self,
                 intent_classifier_file=None,
                 entity_extractor_file=None,
                 feature_extractor_file=None,
                 entity_synonyms=None, **kwargs):
        if entity_extractor_file:
            self.extractor = named_entity_extractor(entity_extractor_file)  # ,metadata["feature_extractor"])
        with open(intent_classifier_file, 'rb') as f:
            self.classifier = cloudpickle.load(f)
        self.featurizer = MITIEFeaturizer(feature_extractor_file)
        self.tokenizer = MITIETokenizer()
        self.ent_synonyms = None
        if entity_synonyms:
            self.ent_synonyms = Interpreter.load_synonyms(entity_synonyms)

    def get_entities(self, tokens):
        d = {}
        entities = self.extractor.extract_entities(tokens)
        for e in entities:
            _range = e[0]
            d[e[1]] = " ".join(tokens[i] for i in _range)
        return d

    def get_intent(self, text):
        """Returns the most likely intent and its probability for the input text.

        :param text: text to classify
        :return: tuple of most likely intent name and its probability"""
        if self.classifier:
            X = self.featurizer.create_bow_vecs([text])
            intent_ids, probabilities = self.classifier.predict(X)
            intents = self.classifier.transform_labels_num2str(intent_ids)
            intent, score = intents[0], probabilities[0]
        else:
            intent, score = "None", 0.0

        return intent, score

    def parse(self, text):
        tokens = self.tokenizer.tokenize(text)
        intent, probability = self.get_intent(tokens)
        entities = self.get_entities(tokens)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
