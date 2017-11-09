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
from sklearn.model_selection import cross_val_score

from rasa_core.policies import Policy

logger = logging.getLogger(__name__)


class SklearnPolicy(Policy):
    """Use an sklearn classifier to train a policy.

    """
    def __init__(
            self,
            featurizer=None,
            max_history=None,
            estimator=LogisticRegression(),
            cv=None,
            param_grid=None,
            scoring='accuracy',
    ):
        self.featurizer = featurizer
        self.max_history = max_history
        self.estimator = estimator
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring

    def model_architecture(self, *args, **kwargs):
        return self.estimator

    def _preprocess_data(self, X, y=None):
        Xt = X.reshape(X.shape[0], -1)
        if y is not None:
            return Xt, y
        return Xt

    def _fit_and_score(self, estimator, X, y):
        scores = cross_val_score(
            estimator, X, y, cv=self.cv, scoring=self.scoring)
        valid_score = np.mean(scores)
        return clone(estimator).fit(X, y), valid_score

    def _search_and_score(self, estimator, X, y, param_grid):
        search = GridSearchCV(
            estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring='accuracy',
            verbose=1,
        )
        search.fit(X, y)
        print("Best params:", search.best_params_)
        return search.best_estimator_, search.best_score_

    def train(self, X, y, domain, **kwargs):
        model = self.model_architecture(domain, **kwargs)
        score = None
        Xt, yt = self._preprocess_data(X, y)

        if self.cv is None:
            model_fit = model.fit(Xt, yt)
        elif self.param_grid is None:
            model_fit, score = self._fit_and_score(model, Xt, yt)
        else:
            model_fit, score = self._search_and_score(
                model, Xt, yt, self.param_grid)

        self.model = model_fit
        logger.info("Done fitting sklearn policy model")
        if score is not None:
            logger.info("Cross validation score: {:.5f}".format(score))

    def continue_training(self, X, y, domain, **kwargs):
        Xt, yt = self._preprocess_data(X, y)
        self.model.partial_fit(Xt, yt)

    def predict_action_probabilities(self, tracker, domain):
        X_feat = self.featurize(tracker, domain)
        Xt = self._preprocess_data(X_feat[np.newaxis])
        y_proba = self.estimator.predict_proba(Xt)
        yp = y_proba[0].tolist()
        # There is no class 1! We need to do this to avoid off-by-one
        yp.insert(1, 0.0)
        return yp

    def persist(self, path):
        if not self.model:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")
            return

        filename = os.path.join(path, 'sklearn_model.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path, featurizer, max_history):
        filename = os.path.join(path, 'sklearn_model.pkl')
        if not os.path.exists(path):
            raise OSError("Failed to load dialogue model. Path {} "
                          "doesn't exist".format(os.path.abspath(filename)))

        with open(filename, 'rb') as f:
            model = pickle.load(f)
        logger.info("Loaded sklearn model")
        return cls(
            featurizer=featurizer,
            max_history=max_history,
            estimator=model,
        )
