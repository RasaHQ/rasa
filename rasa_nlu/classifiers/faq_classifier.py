from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from builtins import zip
import os
import io
from future.utils import PY3
from typing import Any
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.components import Component
from rasa_nlu.featurizers.spacy_featurizer import features_for_sentences
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    import sklearn
    import spacy
    import numpy as np


class FAQClassifierSklearn(Component):
    """FAQ classifier using the sklearn framework"""

    name = "faq_classifier_sklearn"

    number_of_neighbours = 3

    context_provides = {
        "process": ["faq", "faq_ranking"],
    }

    output_provides = ["faq", "faq_ranking"]

    def __init__(self, clf=None, le=None):
        # type: (sklearn.neighbors.KNeighborsClassifier, sklearn.preprocessing.LabelEncoder) -> None
        """Construct a new faq classifier using the sklearn framework."""
        from sklearn.preprocessing import LabelEncoder

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()
        self.clf = clf

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy", "sklearn"]

    def transform_labels_str2num(self, labels):
        # type: (List[Text]) -> np.ndarray
        """Transforms a list of strings into numeric label representation.

        :param labels: List of labels to convert to numeric representation"""

        return self.le.fit_transform(labels)

    def transform_labels_num2str(self, y):
        # type: (np.ndarray) -> np.ndarray
        """Transforms a list of strings into numeric label representation.

        :param y: List of labels to convert to numeric representation"""

        return self.le.inverse_transform(y)

    def train(self, training_data, intent_features, spacy_nlp, num_threads):
        # type: (TrainingData, spacy.language.Language, int) -> None
        """Train the intent classifier on a data set.

        :param num_threads: number of threads used during training time"""
        from sklearn.neighbors import KNeighborsClassifier

        labels = [e["refinement"] for e in training_data.intent_examples if "refinement" in e]
        ex_idx = ["refinement" in e for e in training_data.intent_examples]

        y = self.transform_labels_str2num(labels)

        X = intent_features[ex_idx,:]

        self.clf = KNeighborsClassifier(n_neighbors=self.number_of_neighbours)

        self.clf.fit(X, y)

    def process(self, intent_features):
        # type: (np.ndarray) -> Dict[Text, Any]
        """Returns the most likely intent and its probability for the input text."""

        X = intent_features.reshape(1, -1)
        intent_ids, probabilities = self.predict(X)
        faqs = self.transform_labels_num2str(intent_ids)
        # `predict` returns a matrix as it is supposed to work for multiple examples as well, hence we need to flatten
        faqs, probabilities = faqs.flatten(), probabilities.flatten()
        if faqs.size > 0 and probabilities.size > 0:
            ranking = list(zip(list(faqs), list(probabilities)))
            return {
                "faq": {
                    "name": faqs[0],
                    "confidence": probabilities[0],
                },
                "faq_ranking": [{"name": intent, "confidence": score} for intent, score in ranking]
            }
        else:
            return {"faq": None, "faq_ranking": []}

    def predict_prob(self, X):
        # type: (np.ndarray) -> np.ndarray
        """Given a bow vector of an input text, predict the intent label. Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        return self.clf.predict_proba(X)

    def predict(self, X):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Given a bow vector of an input text, predict most probable label. Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        import numpy as np

        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    @classmethod
    def load(cls, model_dir, faq_classifier_sklearn):
        # type: (Text, Text) -> SklearnIntentClassifier
        import cloudpickle

        if model_dir and faq_classifier_sklearn:
            classifier_file = os.path.join(model_dir, faq_classifier_sklearn)
            with io.open(classifier_file, 'rb') as f:
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return FAQClassifierSklearn()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        import cloudpickle

        classifier_file = os.path.join(model_dir, "faq_classifier_knn.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "faq_classifier_sklearn": "faq_classifier_knn.pkl"
        }
