import cloudpickle
import spacy

from rasa_nlu import Interpreter
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer


class SpacySklearnInterpreter(Interpreter):
    def __init__(self, entity_extractor=None, intent_classifier=None, language_name='en', **kwargs):
        self.extractor = None
        self.classifier = None
        self.nlp = spacy.load(language_name, parser=False, entity=False, matcher=False)
        self.featurizer = SpacyFeaturizer(self.nlp)
        if intent_classifier:
            with open(intent_classifier, 'rb') as f:
                self.classifier = cloudpickle.load(f)
        if entity_extractor:
            self.extractor = SpacyEntityExtractor(self.nlp, entity_extractor)

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

    def get_entities(self, text):
        if self.extractor:
            return self.extractor.extract_entities(self.nlp, text)
        return []

    def parse(self, text):
        """Parse the input text, classify it and return an object containing its intent and entities."""

        intent, probability = self.get_intent(text)
        entities = self.get_entities(text)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
