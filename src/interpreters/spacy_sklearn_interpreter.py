import cloudpickle
import spacy

from rasa_nlu import Interpreter
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer


class SpacySklearnInterpreter(Interpreter):
    def __init__(self, entity_extractor=None, intent_classifier=None, language_name='en', **kwargs):
        self.nlp = spacy.load(language_name, parser=False, entity=False, matcher=False)
        self.featurizer = SpacyFeaturizer(self.nlp)
        with open(intent_classifier, 'rb') as f:
            self.classifier = cloudpickle.load(f)

        self.extractor = SpacyEntityExtractor(self.nlp, entity_extractor)

    def get_intent(self, text):
        """Returns the most likely intent and its probability for the input text.

        :param text: text to classify
        :return: tuple of most likely intent name and its probability"""

        X = self.featurizer.create_bow_vecs([text])
        intent_ids, probabilities = self.classifier.predict(X)
        intents = self.classifier.transform_labels_num2str(intent_ids)
        return intents[0], probabilities[0]

    def parse(self, text):
        """Parse the input text, classify it and return an object containing its intent and entities."""

        intent, probability = self.get_intent(text)
        entities = self.extractor.extract_entities(self.nlp, text)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
