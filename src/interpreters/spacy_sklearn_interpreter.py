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
    @staticmethod
    def load(meta, nlp, featurizer=None):
        """
        :type meta: rasa_nlu.model.Metadata
        :type nlp: spacy.language.Language
        :type featurizer: None or rasa_nlu.featurizers.spacy_featurizer.SpacyFeaturizer
        :rtype: MITIEInterpreter
        """
        if meta.entity_extractor_path:
            extractor = SpacyEntityExtractor(nlp, meta.entity_extractor_path)
        else:
            extractor = None
        if meta.intent_classifier_path:
            with open(meta.intent_classifier_path, 'rb') as f:
                classifier = cloudpickle.load(f)
        else:
            classifier = None
        if meta.entity_synonyms_path:
            entity_synonyms = Interpreter.load_synonyms(meta.entity_synonyms_path)
        else:
            entity_synonyms = None

        if featurizer is None:
            featurizer = SpacyFeaturizer(nlp)
        return SpacySklearnInterpreter(
            classifier,
            extractor,
            entity_synonyms,
            featurizer,
            nlp)

    def __init__(self,
                 intent_classifier=None,
                 entity_extractor=None,
                 entity_synonyms=None,
                 featurizer=None,
                 nlp=None):
        self.extractor = entity_extractor
        self.classifier = intent_classifier
        self.ent_synonyms = entity_synonyms
        self.nlp = nlp
        self.featurizer = featurizer

        ensure_proper_language_model(nlp)

    def get_intent(self, doc):
        """Returns the most likely intent and its probability for the input text.

        :param text: text to classify
        :return: tuple of most likely intent name and its probability"""
        if self.classifier:
            X = self.featurizer.features_for_doc(doc).reshape(1, -1)
            intent_ids, probabilities = self.classifier.predict(X)
            intents = self.classifier.transform_labels_num2str(intent_ids)
            intent, score = intents[0], probabilities[0]
        else:
            intent, score = "None", 0.0

        return intent, score

    def get_entities(self, doc):
        if self.extractor:
            return self.extractor.extract_entities(doc)
        return []

    def parse(self, text):
        """Parse the input text, classify it and return an object containing its intent and entities."""
        doc = self.nlp(text)
        intent, probability = self.get_intent(doc)
        entities = self.get_entities(doc)
        if self.ent_synonyms:
            Interpreter.replace_synonyms(entities, self.ent_synonyms)

        return {'text': text, 'intent': intent, 'entities': entities, 'confidence': probability}
