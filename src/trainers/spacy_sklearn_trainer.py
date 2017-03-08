import spacy
from rasa_nlu.utils.spacy import SPACY_BACKEND_NAME

from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.trainers.trainer import Trainer
from rasa_nlu.utils.spacy import ensure_proper_language_model


class SpacySklearnTrainer(Trainer):
    SUPPORTED_LANGUAGES = {"en", "de"}

    name = SPACY_BACKEND_NAME

    def __init__(self, language_name, max_num_threads=1, should_fine_tune_spacy_ner=False):
        super(self.__class__, self).__init__(language_name, max_num_threads)
        self.should_fine_tune_spacy_ner = should_fine_tune_spacy_ner
        self.nlp = self._load_nlp_model(language_name, should_fine_tune_spacy_ner)
        self.featurizer = SpacyFeaturizer(self.nlp)
        ensure_proper_language_model(self.nlp)

    def train_entity_extractor(self, entity_examples):
        self.entity_extractor = SpacyEntityExtractor()
        self.entity_extractor.train(self.nlp, entity_examples, self.should_fine_tune_spacy_ner)

    def _load_nlp_model(self, language_name, should_fine_tune_spacy_ner):
        # If fine tuning is disabled, we do not need to load the spacy entity model
        if should_fine_tune_spacy_ner:
            return spacy.load(language_name, parser=False)
        else:
            return spacy.load(language_name, parser=False, entity=False)

    def train_intent_classifier(self, intent_examples, test_split_size=0.1):
        self.intent_classifier = SklearnIntentClassifier.train(intent_examples,
                                                               self.featurizer,
                                                               self.max_num_threads,
                                                               test_split_size)
