from mitie import *
import cloudpickle
import datetime
import json
import os

from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
from rasa_nlu.trainers.trainer import Trainer
from training_utils import write_training_metadata


class MITIESklearnTrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en"}

    def __init__(self, fe_file, language_name, max_num_threads=1):
        self.name = "mitie_sklearn"
        self.training_data = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.training_data = None
        self.fe_file = fe_file
        self.featurizer = MITIEFeaturizer(self.fe_file)
        self.max_num_threads = max_num_threads
        self.ensure_language_support(language_name)

    def train(self, data, test_split_size=0.1):
        self.training_data = data
        self.train_intent_classifier(data.intent_examples, test_split_size)

        num_entity_examples = len([e for e in data.entity_examples if len(e["entities"]) > 0])
        if num_entity_examples > 0:
            self.entity_extractor = self.train_entity_extractor(data.entity_examples)

    def start_and_end(self, text_tokens, entity_tokens):
        size = len(entity_tokens)
        max_loc = 1 + len(text_tokens) - size
        locs = [i for i in range(max_loc) if text_tokens[i:i + size] == entity_tokens]
        start, end = locs[0], locs[0] + len(entity_tokens)
        return start, end

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
        trainer.num_threads = self.max_num_threads
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

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        self.intent_classifier = SklearnIntentClassifier(max_num_threads=self.max_num_threads)
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
        classifier_file, entity_extractor_file = None, None
        if self.intent_classifier:
            classifier_file = os.path.join(dir_name, "intent_classifier.pkl")
        if self.entity_extractor:
            entity_extractor_file = os.path.join(dir_name, "entity_extractor.dat")

        write_training_metadata(dir_name, timestamp, data_file, self.name, 'en',
                                classifier_file, entity_extractor_file, self.fe_file)

        with open(data_file, 'w') as f:
            f.write(self.training_data.as_json(indent=2))

        if self.intent_classifier:
            with open(classifier_file, 'wb') as f:
                cloudpickle.dump(self.intent_classifier, f)

        if self.entity_extractor:
            self.entity_extractor.save_to_disk(entity_extractor_file)

        if persistor is not None:
            persistor.send_tar_to_s3(dir_name)
