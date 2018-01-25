from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import copy
import os
import sys
import io
from future.utils import PY3
from typing import Any, Optional
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

from rasa_nlu.classifiers.intent_classifier import IntentClassifier
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)

# How many intents are at max put into the output intent ranking, everything else will be cut off
INTENT_RANKING_LENGTH = 10

# We try to find a good number of cross folds to use during intent training, this specifies the max number of folds
MAX_CV_FOLDS = 5

if typing.TYPE_CHECKING:
    import sklearn
    import numpy as np


class TwostageIntentClassifier(IntentClassifier):
    """Intent classifier using the sklearn framework"""

    name = "intent_classifier_twostage"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    def __init__(self, clf_1=None, clf_2=None, le=None):
        # type: (sklearn.model_selection.GridSearchCV, sklearn.model_selection.GridSearchCV, sklearn.preprocessing.LabelEncoder) -> None
        """Construct a new intent classifier using the sklearn framework."""
        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
        super(TwostageIntentClassifier, self).__init__()

        if le is not None:
            self.le = le
        else:
            self.le = LabelEncoder()

        # two stage relevant information
        self.in_groups = None
        self.out_group = None
        self.n_intents = None

        # first and second stage classifier
        self.clf_1 = clf_1

        self.vectorizer = TfidfVectorizer(min_df=1)
        self.clf_2 = clf_2

    def recurse_finder(self, idx, current_list, bool_array):
        import numpy as np
        current_list.append(idx)
        out = bool_array[idx]
        for jdx in np.where(out == True)[0]:
            if jdx not in current_list:
                current_list = self.recurse_finder(jdx, current_list, bool_array)

        return current_list

    def FOF_finder(self, cm_matrix, threshold):
        # friends of friends group finder
        groups = []
        transition = cm_matrix > threshold
        used_idxs = []
        for idx in range(len(cm_matrix)):
            if idx not in used_idxs:
                group = self.recurse_finder(idx, [], transition)
                used_idxs.extend(group)
                groups.append(group)
        return groups

    def train(self, training_data, config, threshold=0.2, l=None, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """Train the intent classifier on a data set.

        :param num_threads: number of threads used during training time"""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_predict
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix, accuracy_score
        from sklearn.utils.multiclass import unique_labels
        from scipy.sparse import hstack, csr_matrix
        import numpy as np

        labels = [e.get("intent") for e in training_data.intent_examples]

        if len(set(labels)) < 2:
            logger.warn("Can not train an intent classifier. Need at least 2 different classes. " +
                        "Skipping training of intent classifier.")
        else:
            y_true = self.transform_labels_str2num(labels)
            assert(training_data.intent_examples[0].text is not None)
            assert(training_data.intent_examples[0].get("text_features") is not None)
            X_1 = np.stack([example.get("text_features") for example in training_data.intent_examples])
            X_2 = [example.text for example in training_data.intent_examples]
            assert(len(X_2) == X_1.shape[0])

            # # # Training, first stage # # #
            cv_splits = max(2, min(MAX_CV_FOLDS, np.min(np.bincount(y_true)) // 5))  # aim for 5 examples in each fold
            inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=11)

            if not self.clf_1:
                # NOTE str(u'linear') is based on this issue in sklearn
                # https://github.com/EducationalTestingService/skll/issues/87
                # should be fixed in next release
                tuned_parameters = [{"C": [1, 2, 5, 10, 20, 100, 1000], "kernel": [str(u'linear')]}]
                self.clf_1 = GridSearchCV(SVC(probability=True, class_weight='balanced'),
                                        param_grid=tuned_parameters, n_jobs=config["num_threads"],
                                        cv=inner_cv, scoring='f1_weighted', verbose=1)

            # nested CV
            self.clf_1.fit(X_1, y_true)
            outer_cv = StratifiedKFold(n_splits=cv_splits, random_state=11, shuffle=True)
            preds = cross_val_predict(self.clf_1, X_1, y_true, cv=outer_cv)

            # # # Cluster based on first stage # # #
            cnf_matrix = confusion_matrix(y_true, preds, labels=np.unique(y_true))
            self.n_intents = cnf_matrix.shape[0]
            # TODO make this more computationally stable -> possibly add diag term ?
            cnf_matrix = cnf_matrix + cnf_matrix.transpose()

            cnf_matrix = cnf_matrix - np.diag(np.diag(cnf_matrix))
            whitener = np.diag((np.sum(cnf_matrix * 1., axis=0))**-0.5)
            cnf_matrix = np.matmul(whitener, np.matmul(cnf_matrix, whitener))

            groups = self.FOF_finder(np.array(cnf_matrix), threshold)
            named_groups = [[self.transform_labels_num2str(idx) for idx in group] for group in groups if len(group)>1]

            # select all groups with more than 1 member
            in_group = [item for sublist in groups if len(sublist) > 1 for item in sublist]
            # select all other groups
            self.out_group = [i for i in range(self.n_intents) if i not in in_group]
            self.in_groups = [g for g in groups if len(g) > 1]
            assert(len(set(self.out_group).intersection(set(in_group))) == 0)

            # do not filter examples based on previous groups
            preds = np.array(preds)[:, np.newaxis]
            y_2 = y_true

            # tfidf space transform
            self.vectorizer.fit(X_2)
            tfidf = self.vectorizer.transform(X_2)
            # note this produces sparse matrices

            # add group label predictions to improve classification quality
            X_2 = hstack((tfidf.copy(), csr_matrix(preds)))

            if not self.clf_2:
                tuned_parameters = [{"C": [1, 2, 5, 10, 20, 100, 1000], "kernel": [str(u'linear')]}]
                self.clf_2 = GridSearchCV(SVC(probability=True, class_weight='balanced'),
                                        param_grid=tuned_parameters, n_jobs=config["num_threads"],
                                        cv=inner_cv, scoring='f1_weighted', verbose=1)

            self.clf_2.fit(X_2, y_2)

            return


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Returns the most likely intent and its probability for the input text."""

        if not self.clf_1 or not self.clf_2:
            # component is either not trained or didn't receive enough training data
            intent = None
            intent_ranking = []
        else:
            X_1 = message.get("text_features").reshape(1, -1)
            X_2 = [message.text]

            intent_ids, probabilities = self.predict(X_1, X_2)
            intents = self.transform_labels_num2str(intent_ids)
            # `predict` returns a matrix as it is supposed
            # to work for multiple examples as well, hence we need to flatten
            intents, probabilities = intents.flatten(), probabilities.flatten()

            if intents.size > 0 and probabilities.size > 0:
                ranking = list(zip(list(intents), list(probabilities)))[:INTENT_RANKING_LENGTH]
                intent = {"name": intents[0], "confidence": probabilities[0]}
                intent_ranking = [{"name": intent_name, "confidence": score} for intent_name, score in ranking]
            else:
                intent = {"name": None, "confidence": 0.0}
                intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
        return message

    def predict_prob(self, X_1, X_2):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """Given a bow vector of an input text and string of text, predict the intent label. Returns probabilities for all labels.

        :param X: bow of input text
        :return: vector of probabilities containing one entry for each label"""

        import numpy as np
        from scipy.sparse import hstack, csr_matrix

        # # # apply stage 1 classifier # # #
        preds = self.clf_1.predict(X_1)

        # # # apply stage 2 classifier or (else) use results from stage 1 classifier # # #
        # NOTE: hard to fit only on elements from in group because then we don't see all possible labels
        if preds in [x for sublist in self.in_groups for x in sublist]:
            logger.debug("Return stage two prediction")
            preds = np.array(preds)[:, np.newaxis]
            tfidf = self.vectorizer.transform(X_2)
            # note this produces sparse matrices

            # add group label predictions to improve classification quality
            X_2 = hstack((tfidf.copy(), csr_matrix(preds)))
            proba = self.clf_2.predict_proba(X_2)

        else:
            logger.debug("Return stage one prediction")
            proba = self.clf_1.predict_proba(X_1)

        assert(len(proba[0,:]) == self.n_intents)
        return proba

    def predict(self, X_1, X_2):
        # type: (np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """Given a bow vector of an input text, predict most probable label. Returns only the most likely label.

        :param X: bow of input text
        :return: tuple of first, the most probable label and second, its probability"""

        import numpy as np

        pred_result = self.predict_prob(X_1, X_2)
        # sort the probabilities retrieving the indices of the elements in sorted order
        sorted_indices = np.fliplr(np.argsort(pred_result, axis=1))
        return sorted_indices, pred_result[:, sorted_indices]
