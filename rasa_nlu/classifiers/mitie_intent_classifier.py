import os

from rasa_nlu.components import Component


class MitieIntentClassifier(Component):
    name = "intent_mitie"

    def __init__(self, clf=None):
        self.clf = clf

    def train(self, training_data, mitie_file, num_threads):
        from mitie import tokenize, text_categorizer_trainer

        trainer = text_categorizer_trainer(mitie_file)
        trainer.num_threads = num_threads
        for example in training_data.intent_examples:
            tokens = tokenize(example["text"])
            trainer.add_labeled_text(tokens, example["intent"])
        self.clf = trainer.train()

    def process(self, tokens, mitie_feature_extractor):
        intent, score = self.clf(tokens, mitie_feature_extractor)
        return {
            "intent": {
                "name": intent,
                "confidence": score,
            }
        }

    @classmethod
    def load(cls, model_dir, intent_classifier):
        from mitie import text_categorizer

        if model_dir and intent_classifier:
            classifier_file = os.path.join(model_dir, intent_classifier)
            classifier = text_categorizer(classifier_file)
            return MitieIntentClassifier(classifier)
        else:
            return MitieIntentClassifier()

    def persist(self, model_dir):
        import os
        if self.clf:
            classifier_file = os.path.join(model_dir, "intent_classifier.dat")
            self.clf.save_to_disk(classifier_file, pure_model=True)
            return {"intent_classifier": "intent_classifier.dat"}
        else:
            return {"intent_classifier": None}
