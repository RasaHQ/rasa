import json
import logging
import typing
import scipy.sparse
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Union
from collections import defaultdict, OrderedDict

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import rasa.shared.utils.io
import rasa.utils.io as io_utils
import rasa.utils.tensorflow.model_data_utils as model_data_utils
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.nlu.constants import ACTION_TEXT, TEXT
from rasa.shared.nlu.training_data.features import Features
from rasa.utils.tensorflow.model_data import Data, FeatureArray
from rasa.utils.tensorflow.constants import SENTENCE

# noinspection PyProtectedMember
from sklearn.utils import shuffle as sklearn_shuffle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import sklearn


class SklearnPolicy(Policy):
    """Use an sklearn classifier to train a policy."""

    DEFAULT_MAX_HISTORY = 5

    @staticmethod
    def _standard_featurizer(
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> MaxHistoryTrackerFeaturizer:
        # Sklearn policy always uses MaxHistoryTrackerFeaturizer
        return MaxHistoryTrackerFeaturizer(
            state_featurizer=SingleStateFeaturizer(), max_history=5
        )

    def __init__(
        self,
        featurizer: Optional[MaxHistoryTrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: int = DEFAULT_MAX_HISTORY,
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
            priority: Policy priority
            max_history: Maximum history of the dialogs.
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
                    f"Passed featurizer of type '{type(featurizer).__name__}', "
                    f"should be MaxHistoryTrackerFeaturizer."
                )
            if not featurizer.max_history:
                raise ValueError(
                    "Passed featurizer without `max_history`, `max_history` should be "
                    "set to a positive integer value."
                )
        else:
            if not max_history:
                raise ValueError(
                    "max_history should be set to a positive integer value."
                )
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority, **kwargs)

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

        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{SklearnPolicy.__name__}' is deprecated and will be removed in "
            "the future. It is recommended to use the 'TEDPolicy' instead."
        )

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

    @staticmethod
    def _fill_in_features_to_max_length(
        features: List[np.ndarray], max_history: int
    ) -> List[np.ndarray]:
        """
        Pad features with zeros to maximum length;
        Args:
            features: list of features for each dialog;
                each feature has shape [dialog_history x shape_attribute]
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

    def _get_features_for_attribute(
        self, attribute_data: Dict[Text, List[FeatureArray]]
    ):
        """Given a list of all features for one attribute, turn it into a numpy array.

        shape_attribute = features[SENTENCE][0][0].shape[-1]
            (Shape of features of one attribute)

        Args:
            attribute_data: all features in the attribute stored in a FeatureArray

        Returns:
            2D np.ndarray with features for an attribute with
                shape [num_dialogs x (max_history * shape_attribute)]
        """
        sentence_features = attribute_data[SENTENCE][0]

        # vstack serves as removing dimension
        if sentence_features.is_sparse:
            sentence_features = [
                scipy.sparse.vstack(value) for value in sentence_features
            ]
            sentence_features = [feature.toarray() for feature in sentence_features]
        else:
            sentence_features = [np.vstack(value) for value in sentence_features]

        # MaxHistoryFeaturizer is always used with SkLearn policy;
        max_history = self.featurizer.max_history
        features = self._fill_in_features_to_max_length(sentence_features, max_history)
        features = [feature.reshape((1, -1)) for feature in features]
        return np.vstack(features)

    def _preprocess_data(self, data: Data) -> np.ndarray:
        """
        Turn data into np.ndarray for sklearn training; dialogue history features
        are flattened.
        Args:
            data: training data containing all the features
        Returns:
            Training_data: shape [num_dialogs x (max_history * all_features)];
            all_features - sum of number of features of
            intent, action_name, entities, forms, slots.
        """
        if TEXT in data or ACTION_TEXT in data:
            raise Exception(
                f"{self.__name__} cannot be applied to text data. "
                f"Try to use TEDPolicy instead. "
            )

        attribute_data = {
            attribute: self._get_features_for_attribute(attribute_data)
            for attribute, attribute_data in data.items()
        }
        # turning it into OrderedDict so that the order of features is the same
        attribute_data = OrderedDict(attribute_data)
        return np.concatenate(list(attribute_data.values()), axis=-1)

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
        tracker_state_features, label_ids, _ = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )
        training_data, zero_state_features = model_data_utils.convert_to_data_format(
            tracker_state_features
        )
        self.zero_state_features = zero_state_features

        self._train_params.update(kwargs)
        model = self.model_architecture(**self._train_params)
        score = None
        # Note: clone is called throughout to avoid mutating default arguments.
        self.label_encoder = clone(self.label_encoder).fit(label_ids)
        X = self._preprocess_data(training_data)
        y = self.label_encoder.transform(label_ids)

        if self.shuffle:
            X, y = sklearn_shuffle(X, y)

        if self.cv is None:
            model = clone(model).fit(X, y)
        else:
            param_grid = self.param_grid or {}
            model, score = self._search_and_score(model, X, y, param_grid)

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
    ) -> PolicyPrediction:
        X = self.featurizer.create_state_features([tracker], domain, interpreter)
        training_data, _ = model_data_utils.convert_to_data_format(
            X, self.zero_state_features
        )
        Xt = self._preprocess_data(training_data)
        y_proba = self.model.predict_proba(Xt)
        return self._prediction(self._postprocess_prediction(y_proba, domain))

    def persist(self, path: Union[Text, Path]) -> None:
        """Persists the policy properties (see parent class for more information)."""
        if self.model:
            self.featurizer.persist(path)

            meta = {"priority": self.priority}
            path = Path(path)

            meta_file = path / "sklearn_policy.json"
            rasa.shared.utils.io.dump_obj_as_json_to_file(meta_file, meta)

            filename = path / "sklearn_model.pkl"
            rasa.utils.io.pickle_dump(filename, self._state)

            zero_features_filename = path / "zero_state_features.pkl"
            io_utils.pickle_dump(zero_features_filename, self.zero_state_features)

        else:
            rasa.shared.utils.io.raise_warning(
                "Persist called without a trained model present. "
                "Nothing to persist then!"
            )

    @classmethod
    def load(
        cls, path: Union[Text, Path], should_finetune: bool = False, **kwargs: Any
    ) -> Policy:
        """See the docstring for `Policy.load`."""
        filename = Path(path) / "sklearn_model.pkl"
        zero_features_filename = Path(path) / "zero_state_features.pkl"
        if not Path(path).exists():
            raise OSError(
                f"Failed to load dialogue model. Path {filename.absolute()} "
                f"doesn't exist."
            )

        featurizer = TrackerFeaturizer.load(path)
        assert isinstance(featurizer, MaxHistoryTrackerFeaturizer), (
            f"Loaded featurizer of type {type(featurizer).__name__}, should be "
            f"MaxHistoryTrackerFeaturizer."
        )

        meta_file = Path(path) / "sklearn_policy.json"
        meta = json.loads(rasa.shared.utils.io.read_file(meta_file))
        zero_state_features = io_utils.pickle_load(zero_features_filename)

        policy = cls(
            featurizer=featurizer,
            priority=meta["priority"],
            zero_state_features=zero_state_features,
            should_finetune=should_finetune,
        )

        state = io_utils.pickle_load(filename)

        vars(policy).update(state)

        logger.info("Loaded sklearn model")
        return policy
