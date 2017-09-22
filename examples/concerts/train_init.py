from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples.concerts.policy import ConcertPolicy
from rasa_core.agent import Agent
from rasa_core.policies.memoization import MemoizationPolicy

if __name__ == '__main__':
    logging.basicConfig(level="INFO")

    training_data_file = 'examples/concerts/data/stories.md'
    model_path = 'examples/concerts/models/policy/init'

    agent = Agent("examples/concerts/concert_domain.yml",
                  policies=[MemoizationPolicy(), ConcertPolicy()])

    agent.train(
            training_data_file,
            augmentation_factor=50,
            max_history=2,
            epochs=500,
            batch_size=10,
            validation_split=0.2
    )

    agent.persist(model_path)
