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
            model.add(LSTM(self.rnn_size))
            model.add(Dense(input_dim=self.rnn_size, units=output_shape[-1]))
        elif len(output_shape) == 2:
            # y is (num examples, max_dialogue_len, num features) so
            # all the outputs from the rnn are used to
            # calculate the loss, therefore a sequence is returned and
            # time distributed layer is used

            # the first value in input_shape is max dialogue_len,
            # it is set to None, to allow dynamic_rnn creation
            # during prediction
            model.add(Masking(mask_value=-1,
                              input_shape=(None, input_shape[1])))
            model.add(LSTM(self.rnn_size, return_sequences=True))
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
