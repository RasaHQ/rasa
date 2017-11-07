from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from builtins import object

from rasa_core.domain import check_domain_sanity
from rasa_core.interpreter import RegexInterpreter

logger = logging.getLogger(__name__)


class PolicyTrainer(object):
    def __init__(self, ensemble, domain, featurizer):
        self.domain = domain
        self.ensemble = ensemble
        self.featurizer = featurizer

    def train(self, filename=None, max_history=3,
              augmentation_factor=20, max_training_samples=None,
              max_number_of_trackers=2000, remove_duplicates=True, **kwargs):
        """Trains a policy on a domain using training data from a file.

        :param augmentation_factor: how many stories should be created by
                                    randomly concatenating stories
        :param filename: story file containing the training conversations
        :param max_history: number of past actions to consider for the
                            prediction of the next action
        :param max_training_samples: specifies how many training samples to
                                     train on - `None` to use all examples
        :param max_number_of_trackers: limits the tracker generation during
                                       story file parsing - `None` for unlimited
        :param remove_duplicates: remove duplicates from the training set before
                                  training
        :param kwargs: additional arguments passed to the underlying ML trainer
                       (e.g. keras parameters)
        :return: trained policy
        """

        logger.debug("Policy trainer got kwargs: {}".format(kwargs))
        check_domain_sanity(self.domain)

        X, y = self._prepare_training_data(filename, max_history,
                                           augmentation_factor,
                                           max_training_samples,
                                           max_number_of_trackers,
                                           remove_duplicates)

        self.ensemble.train(X, y, self.domain, self.featurizer, **kwargs)

    def _prepare_training_data(self, filename, max_history, augmentation_factor,
                               max_training_samples=None,
                               max_number_of_trackers=2000,
                               remove_duplicates=True):
        """Reads training data from file and prepares it for the training."""

        from rasa_core.training_utils import extract_training_data_from_file

        if filename:
            X, y = extract_training_data_from_file(
                    filename,
                    augmentation_factor=augmentation_factor,
                    max_history=max_history,
                    remove_duplicates=remove_duplicates,
                    domain=self.domain,
                    featurizer=self.featurizer,
                    interpreter=RegexInterpreter(),
                    max_number_of_trackers=max_number_of_trackers)
            if max_training_samples is not None:
                X = X[:max_training_samples, :]
                y = y[:max_training_samples]
        else:
            X = np.zeros((0, self.domain.num_features))
            y = np.zeros(self.domain.num_actions)
        return X, y
