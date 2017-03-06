import datetime
import json

from mitie import *

from rasa_nlu.trainers import mitie_trainer_utils
from rasa_nlu.trainers.trainer import Trainer
from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME
from training_utils import write_training_metadata


class MITIETrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en"}

    def __init__(self, fe_file, language_name, max_num_threads=1):
        super(self.__class__, self).__init__(language_name, max_num_threads)
        self.fe_file = fe_file

    def train_entity_extractor(self, entity_examples):
        self.entity_extractor = mitie_trainer_utils.train_entity_extractor(entity_examples,
                                                                           self.fe_file,
                                                                           self.max_num_threads)

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        trainer = text_categorizer_trainer(self.fe_file)
        trainer.num_threads = self.max_num_threads
        for example in intent_examples:
            tokens = tokenize(example["text"])
            trainer.add_labeled_text(tokens, example["intent"])
        self.intent_classifier = trainer.train()

    def persist(self, path, persistor=None, create_unique_subfolder=True):
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        if create_unique_subfolder:
            dir_name = os.path.join(path, "model_" + timestamp)
            os.makedirs(dir_name)
        else:
            dir_name = path

        data_file = os.path.join(dir_name, "training_data.json")
        entity_synonyms_file = os.path.join(dir_name, "index.json") if self.training_data.entity_synonyms else None
        classifier_file, entity_extractor_file = None, None
        if self.intent_classifier:
            classifier_file = os.path.join(dir_name, "intent_classifier.dat")
        if self.entity_extractor:
            entity_extractor_file = os.path.join(dir_name, "entity_extractor.dat")

        write_training_metadata(dir_name, timestamp, data_file, MITIE_BACKEND_NAME, 'en',
                                classifier_file, entity_extractor_file, entity_synonyms_file, self.fe_file)

        with open(data_file, 'w') as f:
            f.write(self.training_data.as_json(indent=2))

        self.intent_classifier.save_to_disk(classifier_file, pure_model=True)

        if self.entity_extractor:
            self.entity_extractor.save_to_disk(entity_extractor_file, pure_model=True)
        if self.training_data.entity_synonyms:
            with open(entity_synonyms_file, 'w') as f:
                json.dump(self.training_data.entity_synonyms, f)

        if persistor is not None:
            persistor.send_tar_to_s3(dir_name)
        return dir_name
