import logging
from mitie import *
import os
import datetime
import json

from rasa_nlu.trainers.trainer import Trainer
from training_utils import write_training_metadata
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


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

        num_entity_examples = len([e for e in data.entity_examples if len(e["entities"]) > 0])
        if num_entity_examples > 0:
            self.entity_extractor = self.train_entity_extractor(data.entity_examples)

    @classmethod
    def find_entity(cls, ent, text):
        tk = MITIETokenizer()
        tokens, offsets = tk.tokenize_with_offsets(text)
        if ent["start"] not in offsets:
            message = u"invalid entity {0} in example {1}:".format(ent, text) + \
                u" entities must span whole tokens"
            raise ValueError(message)
        start = offsets.index(ent["start"])
        _slice = text[ent["start"]:ent["end"]]
        val_tokens = tokenize(_slice)
        end = start + len(val_tokens)
        return start, end

    def train_entity_extractor(self, entity_examples):
        trainer = ner_trainer(self.fe_file)
        for example in entity_examples:
            text = example["text"]
            tokens = tokenize(text)
            sample = ner_training_instance(tokens)
            for ent in example["entities"]:
                start, end = self.find_entity(ent, text)
                sample.add_entity(xrange(start, end), ent["entity"])

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

        classifier_file, entity_extractor_file = None, None
        if self.intent_classifier:
            classifier_file = os.path.join(dirname, "intent_classifier.dat")
        if self.entity_extractor:
            entity_extractor_file = os.path.join(dirname, "entity_extractor.dat")

        write_training_metadata(dirname, tstamp, data_file, self.name, 'en',
                                classifier_file, entity_extractor_file, self.fe_file)

        with open(data_file, 'w') as f:
            f.write(self.training_data.as_json(indent=2))

        self.intent_classifier.save_to_disk(classifier_file, pure_model=True)

        if self.entity_extractor:
            self.entity_extractor.save_to_disk(entity_extractor_file, pure_model=True)

        if persistor is not None:
            persistor.send_tar_to_s3(dirname)
