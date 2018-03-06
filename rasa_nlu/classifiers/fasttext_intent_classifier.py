from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future.utils import PY3

import logging
import typing
from builtins import zip
import os
import io
from typing import Any, Optional
from typing import List
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)

# How many intents are at max put into the output intent ranking, everything else will be cut off
INTENT_RANKING_LENGTH = 10

class FastTextIntentClassifier(Component):
    """Intent classifier using the sklearn framework
    Here are the references for later use and specs adds
    https://github.com/facebookresearch/fastText/blob/master/python/fastText/FastText.py"""


    name = "intent_classifier_fasttext"

    provides = ["intent", "intent_ranking"]

    # requires = ["text_features"]

    def __init__(self, clf=None, le=None):
        # type: (fastText) -> None
        """Construct a new intent classifier using the fasttext framework."""

        self.clf = clf

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "fastText"]

    def transform_labels(self, labels):
        # type: (List[Text]) -> np.ndarray
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation"""

        return [x.replace('__label__', '') for x in labels]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train the intent classifier on a data set.

        :param num_threads: number of threads used during training time"""

        classifier_file = "projects/default/model.ftz"

        # self.clf = load_model(classifier_file)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        import cloudpickle

        classifier_file = os.path.join(model_dir, "intent_classifier.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_classifier_fasttext": "intent_classifier.pkl"
        }


    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> SklearnIntentClassifier
        import cloudpickle
        from fastText import load_model

        model_file = "projects/default/model.ftz"

        if model_dir and model_metadata.get("intent_classifier_fasttext"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_classifier_fasttext"))

            with io.open(classifier_file, 'rb') as f:  # pragma: no test
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    a = cloudpickle.load(f)
                    a.clf = load_model(model_file)
                    return a
        else:
            return cls()


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        if not self.clf:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            nb_labels = len(self.clf.get_labels())
            intent_ids, probabilities = self.clf.predict(message.text, k=nb_labels)
            intents = self.transform_labels(intent_ids)
            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten

            if len(intents) > 0 and len(probabilities) > 0:
                ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
                intent = {"name": intents[0], "confidence": probabilities[0]}
                intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
