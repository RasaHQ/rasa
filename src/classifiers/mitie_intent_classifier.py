from mitie import text_categorizer_trainer, tokenize, text_categorizer


class MITIEIntentClassifier(object):
    def __init__(self, clf=None):
        self.clf = clf

    @staticmethod
    def train(intent_examples, fe_file, max_num_threads):
        trainer = text_categorizer_trainer(fe_file)
        trainer.num_threads = max_num_threads
        for example in intent_examples:
            tokens = tokenize(example["text"])
            trainer.add_labeled_text(tokens, example["intent"])
        intent_classifier = trainer.train()
        return MITIEIntentClassifier(intent_classifier)

    @staticmethod
    def load(path):
        if path:
            classifier = text_categorizer(path)
            return MITIEIntentClassifier(classifier)
        else:
            return None

    def persist(self, dir_name):
        import os

        classifier_file = os.path.join(dir_name, "intent_classifier.dat")
        self.clf.save_to_disk(classifier_file, pure_model=True)
        return {
            "intent_classifier": "intent_classifier.dat"
        }
