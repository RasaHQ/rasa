import logging


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
        self.train_intent_classifier(data.intent_examples, test_split_size)

        if self.training_data.num_entity_examples > 0:
            self.train_entity_extractor(data.entity_examples)
