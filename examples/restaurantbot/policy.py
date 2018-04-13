from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.policies.keras_policy import KerasPolicy

logger = logging.getLogger(__name__)


class RestaurantPolicy(KerasPolicy):
    def model_architecture(self, input_shape, output_shape):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        from keras.models import Sequential
        from keras.layers import \
            Masking, LSTM, Dense, TimeDistributed, Activation

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        model = Sequential()

        if len(output_shape) == 1:
            model.add(Masking(mask_value=-1, input_shape=input_shape))
            model.add(LSTM(n_hidden))
            model.add(Dense(input_dim=n_hidden, units=output_shape[-1]))
        elif len(output_shape) == 2:
            model.add(Masking(mask_value=-1,
                              input_shape=(None, input_shape[1])))
            model.add(LSTM(n_hidden, return_sequences=True))
            model.add(TimeDistributed(Dense(units=output_shape[-1])))
        else:
            raise ValueError("Cannot construct the model because"
                             "length of output_shape = {} "
                             "should be 1 or 2."
                             "".format(len(output_shape)))

        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model
