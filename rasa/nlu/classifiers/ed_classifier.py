import logging
import os
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import numpy as np

import rasa.utils.io as io_utils
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import TEXT
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message


#################
import requests


print('ED CLASSIFIER v2')

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn

SKLEARN_MODEL_FILE_NAME = "intent_classifier_sklearn.pkl"


def _sklearn_numpy_warning_fix():
    """Fixes unecessary warnings emitted by sklearns use of numpy.

    Sklearn will fix the warnings in their next release in ~ August 2018.

    based on https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array"""
    import warnings

    warnings.filterwarnings(module='sklearn*', action='ignore',
                            category=DeprecationWarning)


class flask_serving_classifier(IntentClassifier):
    """Intent classifier using the sklearn framework"""

    name = "flask_serving_classifier"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    def __init__(self,
                 component_config=None,  # type: Dict[Text, Any]
                 le=None  # type: sklearn.preprocessing.LabelEncoder
                 ):
        # type: (...) -> None
        """Construct a new intent classifier using the sklearn framework."""
        from sklearn.preprocessing import LabelEncoder

        super(flask_serving_classifier, self).__init__(component_config)

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()
        _sklearn_numpy_warning_fix()

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["sklearn"]

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

    def train(self, training_data, cfg, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """Train the intent classifier on a data set."""
        logger.warn("ED CLASSIFIER TRAIN")
        print('ED CLASSIFIER PRINT TRAIN!')
        num_threads = kwargs.get("num_threads", 1)

        labels = [e.get("intent")
                  for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. "
                        "Need at least 2 different classes. "
                        "Skipping training of intent classifier.")
        else:
#             X = np.stack([example.get("text_features")
#                           for example in training_data.intent_examples])

#             attrs = vars(training_data.intent_examples[0])
#             print(', '.join("%s: %s" % item for item in attrs.items()))
#             print('ED TRAIN DATA:', training_data.intent_examples[0])

            X = [i.get(TEXT) for i in training_data.intent_examples]
            eqa_idx = labels.index('EQA_DATA')
            eqa_content = X[eqa_idx]
            print(labels[eqa_idx])
            # Remove the Label and content for EQA Module
            del labels[eqa_idx]
            del X[eqa_idx]
            print(eqa_content)
            y = self.transform_labels_str2num(labels).tolist()
            categories = [i for i in set(y)]
            host = '127.0.0.1'
            port = 9501
            url = f'http://{host}:{port}/train'
            data = {'text': X, 'labels': y, 'unique_labels': categories}
            # print('ED DATA', data)
            tr = requests.put(url, json=data)  ###train
            print(tr.json())

            # self.clf = self._create_classifier(num_threads, y)

            # self.clf.fit(X, y)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its probability for a message."""
        logger.warn("ED CLASSIFIER PROCESS MESSAGE:")
        print('FLASK PROCESS PRINT')
        print('ED message', message.get(TEXT))
        X = [message.get(TEXT)] # this inputs should be a list
        intent_ids, probabilities = self.predict(X)
        intents = self.transform_labels_num2str(np.ravel(intent_ids))
        # `predict` returns a matrix as it is supposed
        # to work for multiple examples as well, hence we need to flatten
        probabilities = probabilities.flatten()
        INTENT_RANKING_LENGTH = 10
        if intents.size > 0 and probabilities.size > 0:
            ranking = list(zip(list(intents),
                               list(probabilities)))[:INTENT_RANKING_LENGTH]

            intent = {"name": intents[0], "confidence": probabilities[0]}

            intent_ranking = [{"name": intent_name, "confidence": score}
                              for intent_name, score in ranking]
        else:
            intent = {"name": None, "confidence": 0.0}
            intent_ranking = []
        print(intent)
        print(intent_ranking)
        message.set("intent", intent, add_to_output=True)
        message.set("eqa_response", "This is a dummy response for EQA.", add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def predict_prob(self, X):
        # type: (np.ndarray) -> np.ndarray
        """Given a bow vector of an input text, predict the intent label.

        Return probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""
        data = {'text': X,'labels':[], 'unique_labels':[]}
        host = '127.0.0.1'
        port = 9501
        url = f'http://{host}:{port}/predict'
        pred = requests.post(url, json=data)
        out = np.array(pred.json()['prediction'])
        return out

    def predict(self, X):
        # type: (np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Given a bow vector of an input text, predict most probable label.

        Return only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second,
                 its probability."""

        pred_result = self.predict_prob(X)
        # sort the probabilities retrieving the indices of
        # the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory."""

        classifier_file_name = file_name + "_classifier.pkl"
        encoder_file_name = file_name + "_encoder.pkl"
        if self.le:
            io_utils.json_pickle(
                os.path.join(model_dir, encoder_file_name), self.le.classes_
            )
        return {"classifier": classifier_file_name, "encoder": encoder_file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["flask_serving_classifier"] = None,
        **kwargs: Any,
    ) -> "flask_serving_classifier":
        from sklearn.preprocessing import LabelEncoder

        encoder_file = os.path.join(model_dir, meta.get("encoder"))

        if os.path.exists(encoder_file):
            classes = io_utils.json_unpickle(encoder_file)
            encoder = LabelEncoder()
            encoder.classes_ = classes
            return cls(meta, encoder)
        else:
            return cls(meta)
