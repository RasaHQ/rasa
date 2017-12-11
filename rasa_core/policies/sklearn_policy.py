from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import pickle
import warnings

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle as sklearn_shuffle

from rasa_core.policies import Policy

logger = logging.getLogger(__name__)


class SklearnPolicy(Policy):
    """Use an sklearn classifier to train a policy.

    Supports cross validation and grid search.

    :param sklearn.base.BaseEstimator model:
      The sklearn model or model pipeline.

    :param cv:
      If *cv* is not None, perform a cross validation on the training
      data. *cv* should then conform to the sklearn standard
      (e.g. *cv=5* for a 5-fold cross-validation).

    :param dict param_grid:
      If *param_grid* is not None and *cv* is given, a grid search on
      the given *param_grid* is performed
      (e.g. *param_grid={'n_estimators': [50, 100]}*).

    :param scoring:
      Scoring strategy, using the sklearn standard.

    :param label_encoder:
      Encoder for the labels. Must implement an *inverse_transform*
      method.

    :param bool shuffle:
      Whether to shuffle training data.

    """
    def __init__(
            self,
            featurizer=None,
            max_history=None,
            model=LogisticRegression(),
            cv=None,
            param_grid=None,
            scoring='accuracy',
            label_encoder=LabelEncoder(),
            shuffle=True,
    ):
        self.featurizer = featurizer
        self.max_history = max_history
        self.model = model
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.label_encoder = label_encoder
        self.shuffle = shuffle

        # attributes that need to be restored after loading
        self._pickle_params = [
            'model', 'cv', 'param_grid', 'scoring', 'label_encoder']

    @property
    def _state(self):
        return {attr: getattr(self, attr) for attr in self._pickle_params}

    def model_architecture(self, *args, **kwargs):
        return self.model.set_params(**kwargs)

    def _preprocess_data(self, X, y=None):
        Xt = X.reshape(X.shape[0], -1)
        if y is None:
            return Xt

        yt = self.label_encoder.transform(y)
        return Xt, yt

    def _postprocess_prediction(self, y_proba):
        yp = y_proba[0].tolist()

        # Class 0 or 1 might not be part of the training labels. Since
        # sklearn does not predict labels it has never encountered
        # during training, it is necessary to insert missing classes.
        indices = self.label_encoder.inverse_transform(np.arange(len(yp)))
        y_filled = [0.0 for _ in range(max(indices + 1))]
        for i, pred in zip(indices, yp):
            y_filled[i] = pred
        return y_filled

    def _search_and_score(self, model, X, y, param_grid):
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=self.cv,
            scoring='accuracy',
            verbose=1,
        )
        search.fit(X, y)
        print("Best params:", search.best_params_)
        return search.best_estimator_, search.best_score_

    def _extract_training_data(self, training_data):
        X, y = training_data.X, training_data.y
        if self.shuffle:
            X, y = sklearn_shuffle(X, y)
        return X, y

    def train(self, training_data, domain, **kwargs):
        # Note: clone is called throughout to avoid mutating default
        # arguments.
        X, y = self._extract_training_data(training_data)
        model = self.model_architecture(domain, **kwargs)
        score = None
        self.label_encoder = clone(self.label_encoder).fit(y)
        Xt, yt = self._preprocess_data(X, y)

        if self.cv is None:
            model = clone(model).fit(Xt, yt)
        else:
            param_grid = self.param_grid or {}
            model, score = self._search_and_score(
                model, Xt, yt, param_grid)

        self.model = model
        logger.info("Done fitting sklearn policy model")
        if score is not None:
            logger.info("Cross validation score: {:.5f}".format(score))

    def continue_training(self, training_data, domain, **kwargs):
        X, y = self._extract_training_data(training_data)
        Xt, yt = self._preprocess_data(X, y)
        if not hasattr(self.model, 'partial_fit'):
            raise TypeError("Continuing training is only possible with "
                            "sklearn models that support 'partial_fit'.")
        self.model.partial_fit(Xt, yt)

    def predict_action_probabilities(self, tracker, domain):
        X_feat = self.featurize(tracker, domain)
        Xt = self._preprocess_data(X_feat[np.newaxis])
        y_proba = self.model.predict_proba(Xt)
        return self._postprocess_prediction(y_proba)

    def persist(self, path):
        if not self.model:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")
            return

        filename = os.path.join(path, 'sklearn_model.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self._state, f)

    @classmethod
    def load(cls, path, featurizer, max_history):
        filename = os.path.join(path, 'sklearn_model.pkl')
        if not os.path.exists(path):
            raise OSError("Failed to load dialogue model. Path {} "
                          "doesn't exist".format(os.path.abspath(filename)))

        with open(filename, 'rb') as f:
            state = pickle.load(f)

        logger.info("Loaded sklearn model")
        policy = cls(
            featurizer=featurizer,
            max_history=max_history,
        )
        vars(policy).update(state)
        return policy
