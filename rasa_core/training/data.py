from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from rasa_core import utils


class DialogueTrainingData(object):
    def __init__(self, X, y, true_length=None):
        self.X = X
        self.y = y
        self.true_length = true_length

    def limit_training_data_to(self, max_samples):
        self.X = self.X[:max_samples]
        self.y = self.y[:max_samples]
        self.true_length = self.true_length[:max_samples]

    def is_empty(self):
        return utils.is_training_data_empty(self.X)

    def max_history(self):
        return self.X.shape[1]

    def num_examples(self):
        return len(self.y)

    def shuffled_X_y(self):
        idx = np.arange(self.num_examples())
        np.random.shuffle(idx)
        shuffled_X = self.X[idx]
        shuffled_y = self.y[idx]
        return shuffled_X, shuffled_y

    def random_samples(self, num_samples):
        padding_idx = np.random.choice(range(self.num_examples()),
                                       replace=False,
                                       size=min(num_samples,
                                                self.num_examples()))

        return self.X[padding_idx], self.y[padding_idx]

    def append(self, X, y):
        self.X = np.vstack((self.X, X))
        self.y = np.vstack((self.y, y))

    @classmethod
    def empty(cls, domain):
        # TODO is it correct?
        X = np.zeros((0, domain.num_features))
        y = np.zeros((0, domain.num_actions))
        return cls(X, y, {})
