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
        self.nlp = spacy.load(self.language_name, parser=False, entity=False)
        self.featurizer = SpacyFeaturizer(self.nlp)
        self.intent_classifier = None
        self.entity_extractor = None

    def train(self, data, test_split_size=0.1):
        self.training_data = data
        self.train_intent_classifier(data.intent_examples, test_split_size)

        num_entity_examples = len([e for e in data.entity_examples if len(e["entities"]) > 0])
        if num_entity_examples > 0:
            self.train_entity_extractor(data.entity_examples)

    def train_entity_extractor(self, entity_examples):
        self.entity_extractor = SpacyEntityExtractor()
        self.entity_extractor.train(self.nlp, entity_examples)

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        self.intent_classifier = SklearnIntentClassifier()
        labels = [e["intent"] for e in intent_examples]
        sentences = [e["text"] for e in intent_examples]
        y = self.intent_classifier.transform_labels_str2num(labels)
        X = self.featurizer.create_bow_vecs(sentences)
        self.intent_classifier.train(X, y, test_split_size)

    def persist(self, path, persistor=None, create_unique_subfolder=True):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        if create_unique_subfolder:
            dir_name = os.path.join(path, "model_" + timestamp)
            os.mkdir(dir_name)
        else:
            dir_name = path

        data_file = os.path.join(dir_name, "training_data.json")
        classifier_file, ner_dir = None, None
        if self.intent_classifier:
            classifier_file = os.path.join(dir_name, "intent_classifier.pkl")
        if self.entity_extractor:
            ner_dir = os.path.join(dir_name, 'ner')
            if not os.path.exists(ner_dir):
                os.mkdir(ner_dir)
            entity_extractor_config_file = os.path.join(ner_dir, "config.json")
            entity_extractor_file = os.path.join(ner_dir, "model")

        write_training_metadata(dir_name, timestamp, data_file, self.name, self.language_name,
                                classifier_file, ner_dir)

        with open(data_file, 'w') as f:
            f.write(self.training_data.as_json(indent=2))
        if self.intent_classifier:
            with open(classifier_file, 'wb') as f:
                cloudpickle.dump(self.intent_classifier, f)
        if self.entity_extractor:
            with open(entity_extractor_config_file, 'w') as f:
                json.dump(self.entity_extractor.ner.cfg, f)

            self.entity_extractor.ner.model.dump(entity_extractor_file)

        if persistor is not None:
            persistor.send_tar_to_s3(dir_name)
