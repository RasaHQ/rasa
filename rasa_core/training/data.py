from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from rasa_core import utils


class DialogueTrainingData(object):
    def __init__(self, X, y, metadata=None):
        self.X = X
        self.y = y
        self.metadata = metadata if metadata else {}

    def limit_training_data_to(self, max_samples):
        self.X = self.X[:max_samples, :]
        self.y = self.y[:max_samples]

    def is_empty(self):
        return utils.is_training_data_empty(self.X)

    def max_history(self):
        return self.X.shape[1]

    def num_examples(self):
        return len(self.y)

    def shuffled(self, domain):
        y_one_hot = self.y_as_one_hot(domain)
        idx = np.arange(self.num_examples())
        np.random.shuffle(idx)
        shuffled_X = self.X[idx, :, :]
        shuffled_y = y_one_hot[idx, :]
        return shuffled_X, shuffled_y

    def y_as_one_hot(self, domain):
        y_one_hot = np.zeros((self.num_examples(), domain.num_actions))
        y_one_hot[np.arange(self.num_examples()), self.y] = 1
        return y_one_hot

    def random_samples(self, num_samples):
        padding_idx = np.random.choice(range(self.num_examples()),
                                       replace=False,
                                       size=min(num_samples,
                                                self.num_examples()))

        return self.X[padding_idx, :, :], self.y[padding_idx]

    def reset_metadata(self):
        self.metadata = {}

    def append(self, X, y):
        self.X = np.vstack((self.X, X))
        self.y = np.hstack((self.y, y))

    @classmethod
    def empty(cls, domain):
        X = np.zeros((0, domain.num_features))
        y = np.zeros(domain.num_actions)
        return cls(X, y, {})
