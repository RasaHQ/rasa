from rasa_nlu import Interpreter
import os
import cloudpickle
import spacy
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor


class SpacySklearnInterpreter(Interpreter):

    def __init__(self, model_dir, nlp=None, entity_extractor=None, intent_classifier=None, **kwargs):    
        classifier_file = os.path.join(model_dir, intent_classifier)
        with open(classifier_file, 'rb') as f:
            self.classifier = cloudpickle.load(f)
        extractor_file = os.path.join(model_dir, entity_extractor)
        print(extractor_file)
        self.extractor = SpacyEntityExtractor(nlp, extractor_file)

    def get_intent(self, text, featurizer, nlp):
        X = featurizer.create_bow_vecs([text], nlp)
        return self.classifier.predict(X)[0]

    def parse(self, text, nlp=None, featurizer=None):
        intent = self.get_intent(text, featurizer, nlp)
        entities = self.extractor.extract_entities(nlp, text)

        return {'text': text, 'intent': intent, 'entities': entities}
