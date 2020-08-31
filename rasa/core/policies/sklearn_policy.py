import json
import logging
import os
import typing
from typing import Any, Callable, Dict, List, Optional, Text, Tuple
from collections import defaultdict
import scipy.sparse

import numpy as np
import rasa.utils.io
from rasa.utils.features import Features
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
from rasa.core.domain import Domain, SubState
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates
from rasa.utils.common import raise_warning
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from rasa.nlu.constants import TEXT, ACTION_TEXT
from rasa.utils.tensorflow.constants import SENTENCE
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format
from rasa.utils.tensorflow.model_data import Data

# noinspection PyProtectedMember
from sklearn.utils import shuffle as sklearn_shuffle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


class SklearnPolicy(Policy):
    """Use an sklearn classifier to train a policy."""

    def __init__(
        self,
        featurizer: Optional[MaxHistoryTrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        model: Optional["sklearn.base.BaseEstimator"] = None,
        param_grid: Optional[Dict[Text, List] or List[Dict]] = None,
        cv: Optional[int] = None,
        scoring: Optional[Text or List or Dict or Callable] = "accuracy",
        label_encoder: LabelEncoder = LabelEncoder(),
        shuffle: bool = True,
        zero_state_features: Optional[Dict[Text, List["Features"]]] = None,
        **kwargs: Any,
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
            zero_state_features: Contains default feature values for attributes
        """

        if featurizer:
            if not isinstance(featurizer, MaxHistoryTrackerFeaturizer):
                raise TypeError(
                    "Passed featurizer of type {}, should be "
                    "MaxHistoryTrackerFeaturizer."
                    "".format(type(featurizer).__name__)
                )
        super().__init__(featurizer, priority)

        self.model = model or self._default_model()
        self.cv = cv
        self.param_grid = param_grid
        self.scoring = scoring
        self.label_encoder = label_encoder
        self.shuffle = shuffle

        # attributes that need to be restored after loading
        self._pickle_params = ["model", "cv", "param_grid", "scoring", "label_encoder"]
        self._train_params = kwargs
        self.zero_state_features = zero_state_features or defaultdict(list)

    @staticmethod
    def _default_model() -> Any:
        return LogisticRegression(solver="liblinear", multi_class="auto")

    @property
    def _state(self):
        return {attr: getattr(self, attr) for attr in self._pickle_params}

    def model_architecture(self, **kwargs) -> Any:
        # filter out kwargs that cannot be passed to model
        train_params = self._get_valid_params(self.model.__init__, **kwargs)
        return self.model.set_params(**train_params)

    def _fill_in_features_to_max_length(
        self, features: List[np.ndarray], max_history: int
    ) -> List[np.ndarray]:
        """
        Pad features with zeros to maximum length;
        Args:
            features: list of features for each dialog; each feature has shape [dialog_history x shape_attribute]
            max_history: maximum history of the dialogs
        Returns:
            padded features
        """
        feature_shape = features[0].shape[-1]
        features = [
            feature
            if feature.shape[0] == max_history
            else np.vstack(
                [np.zeros((max_history - feature.shape[0], feature_shape)), feature]
            )
            for feature in features
        ]
        return features

    def _get_features_for_attribute(self, features: SubState, attribute: Text):
        """
        Given a dictionary for one attribute, turn it into a numpy array;
        shape_attribute = features[SENTENCE][0][0].shape[-1] (Shape of features of one attribute)
        Args:
            features: all features in the attribute stored in a np.array;
        Output:
            2D np.ndarray with features for an attribute; shape: [num_dialogs x (max_history * shape_attribute)]
        """
        sentence_features = features[SENTENCE][0]
        if isinstance(sentence_features[0], scipy.sparse.coo_matrix):
            sentence_features = [feature.toarray() for feature in sentence_features]
        # MaxHistoryFeaturizer is always used with SkLearn policy;
        max_history = self.featurizer.max_history
        features = self._fill_in_features_to_max_length(sentence_features, max_history)
        features = [feature.reshape((1, -1)) for feature in features]
        return np.vstack(features)

    def _preprocess_data(self, X: Data) -> np.ndarray:
        """
        Turn data into np.ndarray for sklearn training; dialogue history features
        are flattened.
        Args:
            X: training data containing all the features
        Returns:
            Training_data: shape [num_dialogs x (max_history * all_features)];
            all_features - sum of number of features of intent, action_name, entities, forms, slots.
        """
        if TEXT in X or ACTION_TEXT in X:
            raise Exception(
                f"{self.__name__} cannot be applied to text data. Try to use TEDPolicy instead. "
            )

        attribute_data = {
            attribute: self._get_features_for_attribute(X[attribute], attribute)
            for attribute in X
        }
        attribute_data = [features for key, features in attribute_data.items()]
        return np.concatenate(attribute_data, axis=-1)

    def _search_and_score(self, model, X, y, param_grid) -> Tuple[Any, Any]:
        search = GridSearchCV(
            model, param_grid=param_grid, cv=self.cv, scoring="accuracy", verbose=1
        )
        search.fit(X, y)
        print("Best params:", search.best_params_)
        return search.best_estimator_, search.best_score_

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        # TODO sklearn policy is broken
        X, y = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )
        training_data, zero_state_features = convert_to_data_format(X)
        self.zero_state_features = zero_state_features

        if self.shuffle:
            X, y = sklearn_shuffle(X, y)

        self._train_params.update(kwargs)
        model = self.model_architecture(**self._train_params)
        score = None
        # Note: clone is called throughout to avoid mutating default
        # arguments.
        self.label_encoder = clone(self.label_encoder).fit(y)
        Xt = self._preprocess_data(training_data)
        yt = self.label_encoder.transform(y)

        if self.cv is None:
            model = clone(model).fit(Xt, yt)
        else:
            param_grid = self.param_grid or {}
            model, score = self._search_and_score(model, Xt, yt, param_grid)

        self.model = model
        logger.info("Done fitting sklearn policy model")
        if score is not None:
            logger.info(f"Cross validation score: {score:.5f}")

    def _postprocess_prediction(self, y_proba, domain) -> List[float]:
        yp = y_proba[0].tolist()

        # Some classes might not be part of the training labels. Since
        # sklearn does not predict labels it has never encountered
        # during training, it is necessary to insert missing classes.
        indices = self.label_encoder.inverse_transform(np.arange(len(yp)))
        y_filled = [0.0 for _ in range(domain.num_actions)]
        for i, pred in zip(indices, yp):
            y_filled[i] = pred

        return y_filled

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> List[float]:
        X = self.featurizer.create_state_features([tracker], domain, interpreter)
        training_data, _ = convert_to_data_format(X, self.zero_state_features)
        Xt = self._preprocess_data(training_data)
        y_proba = self.model.predict_proba(Xt)
        return self._postprocess_prediction(y_proba, domain)

    def persist(self, path: Text) -> None:

        if self.model:
            self.featurizer.persist(path)

            meta = {"priority": self.priority}

            meta_file = os.path.join(path, "sklearn_policy.json")
            rasa.utils.io.dump_obj_as_json_to_file(meta_file, meta)

            filename = os.path.join(path, "sklearn_model.pkl")
            rasa.utils.io.pickle_dump(filename, self._state)
            zero_features_filename = os.path.join(path, "zero_state_features.pkl")
            rasa.utils.io.pickle_dump(
                zero_features_filename, self.zero_state_features,
            )
        else:
            raise_warning(
                "Persist called without a trained model present. "
                "Nothing to persist then!"
            )

    @classmethod
    def load(cls, path: Text) -> Policy:
        filename = os.path.join(path, "sklearn_model.pkl")
        zero_features_filename = os.path.join(path, "zero_state_features.pkl")
        if not os.path.exists(path):
            raise OSError(
                "Failed to load dialogue model. Path {} "
                "doesn't exist".format(os.path.abspath(filename))
            )

        featurizer = TrackerFeaturizer.load(path)
        assert isinstance(featurizer, MaxHistoryTrackerFeaturizer), (
            "Loaded featurizer of type {}, should be "
            "MaxHistoryTrackerFeaturizer.".format(type(featurizer).__name__)
        )

        meta_file = os.path.join(path, "sklearn_policy.json")
        meta = json.loads(rasa.utils.io.read_file(meta_file))
        zero_state_features = rasa.utils.io.pickle_load(zero_features_filename)

        policy = cls(
            featurizer=featurizer,
            priority=meta["priority"],
            zero_state_features=zero_state_features,
        )

        state = rasa.utils.io.pickle_load(filename)

        vars(policy).update(state)

        logger.info("Loaded sklearn model")
        return policy
