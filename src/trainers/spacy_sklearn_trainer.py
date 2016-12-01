import spacy
import os
import datetime
import json
import cloudpickle
from rasa_nlu import util
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.trainers.trainer import Trainer
from training_utils import write_training_metadata


class SpacySklearnTrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en", "de"}

    def __init__(self, config, language_name):
        self.ensure_language_support(language_name)
        self.name = "spacy_sklearn"
        self.language_name = language_name
        self.training_data = None
        self.nlp = spacy.load(self.language_name, tagger=False, parser=False, entity=False)
        self.featurizer = SpacyFeaturizer(self.nlp)
        self.intent_classifier = SklearnIntentClassifier()
        self.entity_extractor = SpacyEntityExtractor()

    def train(self, data):
        self.training_data = data
        self.train_entity_extractor(data.entity_examples)
        self.train_intent_classifier(data.intent_examples)

    def train_entity_extractor(self, entity_examples):
        self.entity_extractor.train(self.nlp, entity_examples)

    def train_intent_classifier(self, intent_examples):
        labels = [e["intent"] for e in intent_examples]
        sentences = [e["text"] for e in intent_examples]
        y = self.intent_classifier.transform_labels(labels)
        X = self.featurizer.create_bow_vecs(sentences)
        self.intent_classifier.train(X, y)

    def persist(self, path, persistor=None):

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        dir_name = os.path.join(path, "model_" + timestamp)
        os.mkdir(dir_name)
        data_file = os.path.join(dir_name, "training_data.json")
        classifier_file = os.path.join(dir_name, "intent_classifier.pkl")
        ner_dir = os.path.join(dir_name, 'ner')
        os.mkdir(ner_dir)
        entity_extractor_config_file = os.path.join(ner_dir, "config.json")
        entity_extractor_file = os.path.join(ner_dir, "model")

        write_training_metadata(dir_name, timestamp, data_file, self.name, self.language_name,
                                classifier_file, ner_dir)

        with open(data_file, 'w') as f:
            f.write(self.training_data.as_json(indent=2))
        with open(classifier_file, 'w') as f:
            cloudpickle.dump(self.intent_classifier, f)
        with open(entity_extractor_config_file, 'w') as f:
            json.dump(self.entity_extractor.ner.cfg, f)

        self.entity_extractor.ner.model.dump(entity_extractor_file)

        if persistor is not None:
            persistor.send_tar_to_s3(dirname)
