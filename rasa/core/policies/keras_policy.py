import copy
import json
import logging
import os
import tensorflow as tf
import numpy as np
import warnings
from typing import Any, List, Dict, Text, Optional, Tuple, Union

import rasa.utils.io

from rasa.core.domain import Domain
from rasa.core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer,
)
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
import rasa.utils.common as common_utils
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
from rasa.constants import DOCS_URL_MIGRATION_GUIDE


# there are a number of issues with imports from tensorflow. hence the deactivation
# pytype: disable=import-error
# pytype: disable=module-attr


try:
    import cPickle as pickle
except ImportError:
    import pickle


logger = logging.getLogger(__name__)


class KerasPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    defaults = {
        # Neural Net and training params
        "rnn_size": 32,
        "epochs": 100,
        "batch_size": 32,
        "validation_split": 0.1,
        # set random seed to any int to get reproducible results
        "random_seed": None,
    }

    @staticmethod
    def _standard_featurizer(max_history=None) -> MaxHistoryTrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(
            BinarySingleStateFeaturizer(), max_history=max_history
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        model: Optional[tf.keras.models.Sequential] = None,
        current_epoch: int = 0,
        max_history: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super().__init__(featurizer, priority)

        self._load_params(**kwargs)
        self.model = model

        self.current_epoch = current_epoch

        common_utils.raise_warning(
            "'KerasPolicy' is deprecated and will be removed in version "
            "2.0. Use 'TEDPolicy' instead.",
            category=FutureWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        # filter out kwargs that are used explicitly
        self.rnn_size = config.pop("rnn_size")
        self.epochs = config.pop("epochs")
        self.batch_size = config.pop("batch_size")
        self.validation_split = config.pop("validation_split")
        self.random_seed = config.pop("random_seed")

        self._train_params = config

    @property
    def max_len(self):
        if self.model:
            return self.model.layers[0].batch_input_shape[1]
        else:
            return None

    def model_architecture(
        self, input_shape: Tuple[int, int], output_shape: Tuple[int, Optional[int]]
    ) -> tf.keras.models.Sequential:
        """Build a keras model and return a compiled model."""

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Masking,
            LSTM,
            Dense,
            TimeDistributed,
            Activation,
        )

        # Build Model
        model = Sequential()

        # the shape of the y vector of the labels,
        # determines which output from rnn will be used
        # to calculate the loss
        if len(output_shape) == 1:
            # y is (num examples, num features) so
            # only the last output from the rnn is used to
            # calculate the loss
            model.add(Masking(mask_value=-1, input_shape=input_shape))
            model.add(LSTM(self.rnn_size, dropout=0.2))
            model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))
        elif len(output_shape) == 2:
            # y is (num examples, max_dialogue_len, num features) so
            # all the outputs from the rnn are used to
            # calculate the loss, therefore a sequence is returned and
            # time distributed layer is used

            # the first value in input_shape is max dialogue_len,
            # it is set to None, to allow dynamic_rnn creation
            # during prediction
            model.add(Masking(mask_value=-1, input_shape=(None, input_shape[1])))
            model.add(LSTM(self.rnn_size, return_sequences=True, dropout=0.2))
            model.add(TimeDistributed(Dense(units=output_shape[-1])))
        else:
            raise ValueError(
                "Cannot construct the model because"
                "length of output_shape = {} "
                "should be 1 or 2."
                "".format(len(output_shape))
            )

        model.add(Activation("softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
        )

        if common_utils.obtain_verbosity() > 0:
            model.summary()

        return model

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:

        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)
        # noinspection PyPep8Naming
        shuffled_X, shuffled_y = training_data.shuffled_X_y()

        if self.model is None:
            self.model = self.model_architecture(
                shuffled_X.shape[1:], shuffled_y.shape[1:]
            )

        logger.debug(
            f"Fitting model with {training_data.num_examples()} total samples and a "
            f"validation split of {self.validation_split}."
        )

        # filter out kwargs that cannot be passed to fit
        self._train_params = self._get_valid_params(
            self.model.fit, **self._train_params
        )

        self.model.fit(
            shuffled_X,
            shuffled_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=common_utils.obtain_verbosity(),
            **self._train_params,
        )
        self.current_epoch = self.epochs

        logger.debug("Done fitting Keras Policy model.")

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:

        # noinspection PyPep8Naming
        X = self.featurizer.create_X([tracker], domain)

        y_pred = self.model.predict(X, batch_size=1)

        if len(y_pred.shape) == 2:
            return y_pred[-1].tolist()
        elif len(y_pred.shape) == 3:
            return y_pred[0, -1].tolist()
        else:
            raise Exception("Network prediction has invalid shape.")

    def persist(self, path: Text) -> None:

        if self.model:
            self.featurizer.persist(path)

            meta = {
                "priority": self.priority,
                "model": "keras_model.h5",
                "epochs": self.current_epoch,
            }

            meta_file = os.path.join(path, "keras_policy.json")
            rasa.utils.io.dump_obj_as_json_to_file(meta_file, meta)

            model_file = os.path.join(path, meta["model"])
            # makes sure the model directory exists
            rasa.utils.io.create_directory_for_file(model_file)
            self.model.save(model_file, overwrite=True)

        else:
            logger.debug(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )

    @classmethod
    def load(cls, path: Text) -> "KerasPolicy":
        from tensorflow.keras.models import load_model

        if os.path.exists(path):
            featurizer = TrackerFeaturizer.load(path)
            meta_file = os.path.join(path, "keras_policy.json")
            if os.path.isfile(meta_file):
                meta = json.loads(rasa.utils.io.read_file(meta_file))

                model_file = os.path.join(path, meta["model"])

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = load_model(model_file)

                return cls(
                    featurizer=featurizer,
                    priority=meta["priority"],
                    model=model,
                    current_epoch=meta["epochs"],
                )
            else:
                return cls(featurizer=featurizer)
        else:
            raise Exception(
                "Failed to load dialogue model. Path {} "
                "doesn't exist".format(os.path.abspath(path))
            )


# pytype: enable=import-error
# pytype: disable=module-attr
