class Trainer(object):
    SUPPORTED_LANGUAGES = None

    def __init__(self, name, language_name, max_num_threads=1):
        self.name = name
        self.intent_classifier = None
        self.entity_extractor = None
        self.language_name = language_name
        self.max_num_threads = max_num_threads
        self.training_data = None
        self.ensure_language_support(language_name)

    def ensure_language_support(self, language_name):
        if language_name not in self.SUPPORTED_LANGUAGES:
            raise NotImplementedError("Selected backend currently does not support language '{}' (only '{}')."
                                      .format(language_name, "', '".join(self.SUPPORTED_LANGUAGES)))

    def train(self, data, test_split_size=0.1):
        self.training_data = data
        self.train_intent_classifier(data.intent_examples, test_split_size)

        num_entity_examples = len([e for e in data.entity_examples if len(e["entities"]) > 0])
        if num_entity_examples > 0:
            self.entity_extractor = self.train_entity_extractor(data.entity_examples)
