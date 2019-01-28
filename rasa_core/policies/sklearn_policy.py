import typing

import logging
import numpy as np
import os
import pickle
import warnings
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
# noinspection PyProtectedMember
from sklearn.utils import shuffle as sklearn_shuffle
from typing import Optional, Any, List, Text, Dict, Callable

from rasa_core.domain import Domain
from rasa_core.featurizers import (
    TrackerFeaturizer, MaxHistoryTrackerFeaturizer)
from rasa_core.policies.policy import Policy
from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


class SklearnPolicy(Policy):
    """Use an sklearn classifier to train a policy."""

    def __init__(
        self,
        featurizer: Optional[MaxHistoryTrackerFeaturizer] = None,
        model: 'sklearn.base.BaseEstimator' = LogisticRegression(),
        param_grid: Optional[Dict[Text, List] or List[Dict]] = None,
        cv: Optional[int] = None,
        scoring: Optional[Text or List or Dict or Callable] = 'accuracy',
        label_encoder: LabelEncoder = LabelEncoder(),
        shuffle: bool = True,
        **kwargs: Any
    ) -> None:
        """Create a new sklearn policy.

        Args:
            featurizer: Featurizer used to convert the training data into
                vector format.
            model: The sklearn model or model pipeline.
            param_grid: If *param_grid* is not None and *cv* is given,
                a grid search on the given *param_grid* is performed
                (e.g. *param_grid={'n_estimators': [50, 100]}*).
            cv: If *cv* is not None, perform a cross validation on
                the training data. *cv* should then conform to the
                sklearn standard (e.g. *cv=5* for a 5-fold cross-validation).
            scoring: Scoring strategy, using the sklearn standard.
            label_encoder: Encoder for the labels. Must implement an
                *inverse_transform* method.
            shuffle: Whether to shuffle training data.
        """

        if featurizer:
            if not isinstance(featurizer, MaxHistoryTrackerFeaturizer):
                raise TypeError("Passed featurizer of type {}, should be "
                                "MaxHistoryTrackerFeaturizer."
                                "".format(type(featurizer).__name__))
        super(SklearnPolicy, self).__init__(featurizer)

        self.model = model
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.label_encoder = label_encoder
        self.shuffle = shuffle

        # attributes that need to be restored after loading
        self._pickle_params = [
            'model', 'cv', 'param_grid', 'scoring', 'label_encoder']
        self._train_params = kwargs

    @property
    def _state(self):
        return {attr: getattr(self, attr) for attr in self._pickle_params}

    def model_architecture(self):
        # filter out kwargs that cannot be passed to model
        self._train_params = self._get_valid_params(self.model.__init__,
                                                    **self._train_params)
        return self.model.set_params(**self._train_params)

    def _extract_training_data(self, training_data):
        # transform y from one-hot to num_classes
        X, y = training_data.X, training_data.y.argmax(axis=-1)
        if self.shuffle:
            X, y = sklearn_shuffle(X, y)
        return X, y

    def _preprocess_data(self, X, y=None):
        Xt = X.reshape(X.shape[0], -1)
        if y is None:
            return Xt
        else:
            yt = self.label_encoder.transform(y)
            return Xt, yt

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

    def train(self,
              training_trackers: List[DialogueStateTracker],
              domain: Domain,
              **kwargs: Any
              ) -> None:

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)

        X, y = self._extract_training_data(training_data)
        model = self.model_architecture(**kwargs)
        score = None
        # Note: clone is called throughout to avoid mutating default
        # arguments.
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

    def _postprocess_prediction(self, y_proba, domain):
        yp = y_proba[0].tolist()

        # Some classes might not be part of the training labels. Since
        # sklearn does not predict labels it has never encountered
        # during training, it is necessary to insert missing classes.
        indices = self.label_encoder.inverse_transform(np.arange(len(yp)))
        y_filled = [0.0 for _ in range(domain.num_actions)]
        for i, pred in zip(indices, yp):
            y_filled[i] = pred

        return y_filled

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        X = self.featurizer.create_X([tracker], domain)
        Xt = self._preprocess_data(X)
        y_proba = self.model.predict_proba(Xt)
        return self._postprocess_prediction(y_proba, domain)

    def persist(self, path: Text) -> None:

        if self.model:
            self.featurizer.persist(path)

            filename = os.path.join(path, 'sklearn_model.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(self._state, f)
        else:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")

    @classmethod
    def load(cls, path: Text) -> Policy:
        filename = os.path.join(path, 'sklearn_model.pkl')
        if not os.path.exists(path):
            raise OSError("Failed to load dialogue model. Path {} "
                          "doesn't exist".format(os.path.abspath(filename)))

        featurizer = TrackerFeaturizer.load(path)
        assert isinstance(featurizer, MaxHistoryTrackerFeaturizer), \
            ("Loaded featurizer of type {}, should be "
             "MaxHistoryTrackerFeaturizer.".format(type(featurizer).__name__))

        policy = cls(featurizer=featurizer)

        with open(filename, 'rb') as f:
            state = pickle.load(f)
        vars(policy).update(state)

        logger.info("Loaded sklearn model")
        return policy
