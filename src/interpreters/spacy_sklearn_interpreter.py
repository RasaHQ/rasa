from rasa_nlu import Interpreter
import os
import cloudpickle
import spacy
import json
import codecs

from rasa_nlu import Interpreter
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.utils.spacy import ensure_proper_language_model


class SpacySklearnInterpreter(Interpreter):
    def __init__(self,
                 entity_extractor=None,
                 entity_synonyms=None,
                 intent_classifier=None,
                 nlp=None,
                 **kwargs):
        self.extractor = None
        self.classifier = None
        self.ent_synonyms = None
        self.featurizer = SpacyFeaturizer()
        self.ent_synonyms = Interpreter.load_synonyms(entity_synonyms)
        ensure_proper_language_model(nlp)

        if intent_classifier:
            with open(intent_classifier, 'rb') as f:
                self.classifier = cloudpickle.load(f)
        if entity_extractor:
            self.extractor = SpacyEntityExtractor(nlp, entity_extractor)

    def get_intent(self, text, nlp):
        """Returns the most likely intent and its probability for the input text.

        :param text: text to classify
        :return: tuple of most likely intent name and its probability"""
        if self.classifier:
            X = self.featurizer.create_bow_vecs([text], nlp=nlp)
            intent_ids, probabilities = self.classifier.predict(X)
            intents = self.classifier.transform_labels_num2str(intent_ids)
            intent, score = intents[0], probabilities[0]
        else:
            intent, score = "None", 0.0

        return intent, score

    def get_entities(self, text, nlp):
        if self.extractor:
            return self.extractor.extract_entities(nlp, text)
        return []

    def parse(self, text, nlp=None, featurizer=None):
        """Parse the input text, classify it and return an object containing its intent and entities."""

        intent, probability = self.get_intent(text, nlp)
        entities = self.get_entities(text, nlp)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
