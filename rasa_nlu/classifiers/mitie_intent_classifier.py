from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    import mitie


class MitieIntentClassifier(Component):

    name = "intent_classifier_mitie"

    context_provides = {
        "process": ["intent"],
    }

    output_provides = ["intent"]

    def __init__(self, clf=None):
        self.clf = clf

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    def train(self, training_data, mitie_file, num_threads):
        # type: (TrainingData, Text, Optional[int]) -> None
        import mitie

        trainer = mitie.text_categorizer_trainer(mitie_file)
        trainer.num_threads = num_threads
        for example in training_data.intent_examples:
            tokens = mitie.tokenize(example["text"])
            trainer.add_labeled_text(tokens, example["intent"])
        self.clf = trainer.train()

    def process(self, tokens, mitie_feature_extractor):
        # type: (List[Text], mitie.total_word_feature_extractor) -> Dict[Text, Any]

        intent, score = self.clf(tokens, mitie_feature_extractor)
        return {
            "intent": {
                "name": intent,
                "confidence": score,
            }
        }

    @classmethod
    def load(cls, model_dir, intent_classifier_mitie):
        # type: (Text, Text) -> MitieIntentClassifier
        import mitie

        if model_dir and intent_classifier_mitie:
            classifier_file = os.path.join(model_dir, intent_classifier_mitie)
            classifier = mitie.text_categorizer(classifier_file)
            return MitieIntentClassifier(classifier)
        else:
            return MitieIntentClassifier()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        import os

        if self.clf:
            classifier_file = os.path.join(model_dir, "intent_classifier.dat")
            self.clf.save_to_disk(classifier_file, pure_model=True)
            return {"intent_classifier_mitie": "intent_classifier.dat"}
        else:
            return {"intent_classifier_mitie": None}
