from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples.concerts.policy import ConcertPolicy
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


def run_concertbot_online(input_channel, interpreter):
    training_data_file = 'examples/concerts/data/stories.md'

    agent = Agent("examples/concerts/concert_domain.yml",
                  policies=[MemoizationPolicy(), ConcertPolicy()],
                  interpreter=interpreter)

    agent.train_online(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_concertbot_online(ConsoleInputChannel(), RegexInterpreter())
