import logging
from mitie import *
import os
import datetime
import json

from rasa_nlu.trainers.trainer import Trainer
from training_utils import write_training_metadata


class MITIETrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en"}

    def __init__(self, fe_file, language_name):
        self.name = "mitie"
        self.training_data = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.training_data = None
        self.fe_file = fe_file
        self.ensure_language_support(language_name)

    def train(self, data):
        self.training_data = data
        self.intent_classifier = self.train_intent_classifier(data.intent_examples)
        self.entity_extractor = self.train_entity_extractor(data.entity_examples)

    def find_location_of_entity(self, text_tokens, entity_tokens):
        num_entity_tokens = len(entity_tokens)
        max_loc = 1 + len(text_tokens) - num_entity_tokens
        for i in range(max_loc):
            if text_tokens[i:i + num_entity_tokens] == entity_tokens:
                return i, i + num_entity_tokens
        return None

    def train_entity_extractor(self, entity_examples):
        trainer = ner_trainer(self.fe_file)
        for example in entity_examples:
            tokens = tokenize(example["text"])
            sample = ner_training_instance(tokens)
            for ent in example["entities"]:
                _slice = example["text"][ent["start"]:ent["end"]]
                val_tokens = tokenize(_slice)
                entity_location = self.find_location_of_entity(tokens, val_tokens)
                if entity_location:
                    sample.add_entity(xrange(entity_location[0], entity_location[1]), ent["entity"])
                else:
                    logging.warn("Ignored invalid entity example. Make sure indices are correct. " +
                                 "Text: \"{}\", Invalid entity: \"{}\".".format(example["text"], _slice))
            trainer.add(sample)

        ner = trainer.train()
        return ner

    def train_intent_classifier(self, intent_examples):
        trainer = text_categorizer_trainer(self.fe_file)
        for example in intent_examples:
            tokens = tokenize(example["text"])
            trainer.add_labeled_text(tokens, example["intent"])

        intent_classifier = trainer.train()
        return intent_classifier

    def persist(self, path, persistor=None):
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        dirname = os.path.join(path, "model_" + tstamp)
        os.mkdir(dirname)
        data_file = os.path.join(dirname, "training_data.json")
        classifier_file = os.path.join(dirname, "intent_classifier.dat")
        entity_extractor_file = os.path.join(dirname, "entity_extractor.dat")

        write_training_metadata(dirname, tstamp, data_file, self.name, 'en',
                                classifier_file, entity_extractor_file, self.fe_file)

        with open(data_file, 'w') as f:
            f.write(self.training_data.as_json(indent=2))

        self.intent_classifier.save_to_disk(classifier_file, pure_model=True)
        self.entity_extractor.save_to_disk(entity_extractor_file, pure_model=True)

        if persistor is not None:
            persistor.send_tar_to_s3(dirname)
