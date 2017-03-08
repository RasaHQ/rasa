from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME

from rasa_nlu.classifiers.mitie_intent_classifier import MITIEIntentClassifier
from rasa_nlu.extractors.mitie_entity_extractor import MITIEEntityExtractor
from rasa_nlu.trainers.trainer import Trainer


class MITIETrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en"}

    name = MITIE_BACKEND_NAME

    def __init__(self, fe_file, language_name, max_num_threads=1):
        super(self.__class__, self).__init__(language_name, max_num_threads)
        self.fe_file = fe_file

    def train_entity_extractor(self, entity_examples):
        self.entity_extractor = MITIEEntityExtractor.train(entity_examples, self.fe_file, self.max_num_threads)

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        self.intent_classifier = MITIEIntentClassifier.train(intent_examples, self.fe_file, self.max_num_threads)
