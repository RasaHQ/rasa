from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os
import warnings

import numpy as np
from builtins import str

from rasa_core import utils
from rasa_core.policies import Policy

logger = logging.getLogger(__name__)


class KerasPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    def __init__(self, model=None, graph=None, current_epoch=0,
                 featurizer=None, max_history=None):
        import keras

        super(KerasPolicy, self).__init__(featurizer, max_history)
        if KerasPolicy.is_using_tensorflow() and not graph:
            self.graph = keras.backend.tf.get_default_graph()
        else:
            self.graph = graph
        self.model = model
        self.current_epoch = current_epoch

    @property
    def max_len(self):
        if self.model:
            return self.model.layers[0].batch_input_shape[1]
        else:
            return None

    @staticmethod
    def is_using_tensorflow():
        import keras
        return keras.backend._BACKEND == "tensorflow"

    def predict_action_probabilities(self, tracker, domain):
        x = self.featurize(tracker, domain)
        # we need to add a batch dimension with length 1
        x = x.reshape((1, self.max_len, x.shape[1]))
        if KerasPolicy.is_using_tensorflow() and self.graph is not None:
            with self.graph.as_default():
                y_pred = self.model.predict(x, batch_size=1)
        else:
            y_pred = self.model.predict(x, batch_size=1)
        return y_pred[-1].tolist()

    def _build_model(self, num_features, num_actions, max_history_len):
        warnings.warn("Deprecated, use `model_architecture` instead.",
                      DeprecationWarning, stacklevel=2)
        return

    def model_architecture(self, num_features, num_actions, max_history_len):
        """Build a keras model and return a compiled model.

        :param max_history_len: The maximum number of historical
                                turns used to decide on next action
        """
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # Neural Net and training params
        batch_shape = (None, max_history_len, num_features)
        # Build Model
        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape, dropout=0.2))
        model.add(Dense(input_dim=n_hidden, units=num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model

    def train(self, X, y, domain, **kwargs):
        self.model = self.model_architecture(domain.num_features,
                                             domain.num_actions,
                                             X.shape[1])
        y_one_hot = np.zeros((len(y), domain.num_actions))
        y_one_hot[np.arange(len(y)), y] = 1

        number_of_samples = X.shape[0]
        idx = np.arange(number_of_samples)
        np.random.shuffle(idx)
        shuffled_X = X[idx, :, :]
        shuffled_y = y_one_hot[idx, :]

        validation_split = kwargs.get("validation_split", 0.0)
        logger.info("Fitting model with {} total samples and a validation "
                    "split of {}".format(number_of_samples, validation_split))
        self.model.fit(shuffled_X, shuffled_y, **kwargs)
        self.current_epoch = kwargs.get("epochs", 10)
        logger.info("Done fitting keras policy model")

    def continue_training(self, X, y, domain, **kwargs):
        # fit to one extra example
        y_one_hot = np.zeros((len(y), domain.num_actions))
        y_one_hot[np.arange(len(y)), y] = 1
        self.current_epoch += 1
        self.model.fit(X, y_one_hot,
                       epochs=self.current_epoch + 1,
                       batch_size=1,
                       verbose=0,
                       initial_epoch=self.current_epoch)

    def persist(self, path):
        if self.model:
            arch_file = os.path.join(path, 'keras_arch.json')
            weights_file = os.path.join(path, 'keras_weights.h5')
            utils.create_dir_for_file(weights_file)
            with io.open(arch_file, 'w') as f:
                f.write(str(self.model.to_json()))
            with io.open(os.path.join(path, 'keras_policy.json'), 'w') as f:
                f.write(str(json.dumps({
                    "arch": "keras_arch.json",
                    "weights": "keras_weights.h5",
                    "epochs": self.current_epoch})))
            self.model.save_weights(weights_file, overwrite=True)
        else:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")

    @classmethod
    def _load_model_arch(cls, path, meta):
        from keras.models import model_from_json

        arch_file = os.path.join(path, meta["arch"])
        if os.path.isfile(arch_file):
            with io.open(arch_file) as f:
                model = model_from_json(f.read())
            return model
        else:
            return None

    @classmethod
    def _load_weights_for_model(cls, path, model, meta):
        weights_file = os.path.join(path, meta["weights"])
        if model is not None and os.path.exists(weights_file):
            model.load_weights(weights_file)
        return model

    @classmethod
    def load(cls, path, featurizer, max_history):
        if os.path.exists(path):
            meta_path = os.path.join(path, "keras_policy.json")
            if os.path.isfile(meta_path):
                with io.open(meta_path) as f:
                    meta = json.loads(f.read())
                model_arch = cls._load_model_arch(path, meta)
                return KerasPolicy(
                        cls._load_weights_for_model(path, model_arch, meta),
                        current_epoch=meta["epochs"],
                        max_history=max_history,
                        featurizer=featurizer
                )
            else:
                return KerasPolicy(max_history=max_history,
                                   featurizer=featurizer)
        else:
            raise Exception("Failed to load dialogue model. Path {} "
                            "doesn't exist".format(os.path.abspath(path)))
