from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from builtins import object
from typing import Text, Optional, Any, List

from rasa_core.domain import check_domain_sanity
from rasa_core.training.generator import TrainingsDataGenerator

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.policies.ensemble import PolicyEnsemble
    from rasa_core.training.generator import GeneratorOut


class PolicyTrainer(object):
    def __init__(self, ensemble, domain):
        # type: (PolicyEnsemble, Domain) -> None
        self.domain = domain
        self.ensemble = ensemble

    def train(self,
              resource_name=None,  # type: Optional[Text]
              augmentation_factor=20,  # type: int
              max_training_samples=None,  # type: Optional[int]
              max_number_of_trackers=2000,  # type: int
              remove_duplicates=True,  # type: bool
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Trains a policy on a domain using training data from a file.

        :param resource_name: story file containing the training conversations
        :param augmentation_factor: how many stories should be created by
                                    randomly concatenating stories
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

        training_trackers, events_metadata = self.extract_trackers(
                resource_name,
                self.domain,
                augmentation_factor=augmentation_factor,
                remove_duplicates=remove_duplicates,
                max_number_of_trackers=max_number_of_trackers
        )

        self.ensemble.train(training_trackers, events_metadata, self.domain,
                            max_training_samples=max_training_samples, **kwargs)

    @staticmethod
    def extract_trackers(
            resource_name,  # type: Text
            domain,  # type: Domain
            remove_duplicates=True,  # type: bool
            augmentation_factor=20,  # type: int
            max_number_of_trackers=2000,  # type: int
            tracker_limit=None,  # type: Optional[int]
            use_story_concatenation=True  # type: bool

    ):
        # type: (...) -> GeneratorOut
        if resource_name:

            # TODO pass extract_story_graph as a function or
            # TODO pass graph to agent
            # TODO to support code creating stories
            from rasa_core.training import extract_story_graph
            graph = extract_story_graph(resource_name, domain)

            g = TrainingsDataGenerator(graph, domain,
                                       remove_duplicates,
                                       augmentation_factor,
                                       max_number_of_trackers,
                                       tracker_limit,
                                       use_story_concatenation)
            return g.generate()
        else:
            return [], None

