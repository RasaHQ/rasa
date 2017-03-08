import logging

import datetime
import os

from rasa_nlu.trainers.training_utils import write_training_metadata


class Component(object):
    def tain(self):
        pass

    def process(self):
        pass

    def persist(self):
        pass


class Trainer(object):
    SUPPORTED_LANGUAGES = None

    def __init__(self, language_name, max_num_threads=1):
        self.intent_classifier = None
        self.entity_extractor = None
        self.language_name = language_name
        self.max_num_threads = max_num_threads
        self.training_data = None
        self.nlp = None
        self.ensure_language_support(language_name)
        # self.pipeline = pipeline
        self.fe_file = None

    def ensure_language_support(self, language_name):
        if language_name not in self.SUPPORTED_LANGUAGES:
            supported = "', '".join(self.SUPPORTED_LANGUAGES)
            logging.warn("Selected backend currently does not officially support language " +
                         "'{}' (only '{}'). Things might break!".format(language_name, supported))

    def train_entity_extractor(self, entity_examples):
        raise NotImplementedError()

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        raise NotImplementedError()

    def train(self, data, test_split_size=0.1):
        self.training_data = data

        # for component in self.pipeline:
        #     component.train()

        self.train_intent_classifier(data.intent_examples, test_split_size)

        if self.training_data.num_entity_examples > 0:
            self.train_entity_extractor(data.entity_examples)

    def persist(self, path, persistor=None, create_unique_subfolder=True):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {}

        if create_unique_subfolder:
            dir_name = os.path.join(path, "model_" + timestamp)
            os.makedirs(dir_name)
        else:
            dir_name = path

        metadata.update(self.training_data.persist(dir_name))

        if self.intent_classifier:
            metadata.update(self.intent_classifier.persist(dir_name))

        if self.entity_extractor:
            metadata.update(self.entity_extractor.persist(dir_name))

        write_training_metadata(dir_name, timestamp, self.name, self.language_name,
                                metadata, self.fe_file)

        if persistor is not None:
            persistor.send_tar_to_s3(dir_name)
        return dir_name
