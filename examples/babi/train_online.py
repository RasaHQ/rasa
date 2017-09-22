from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from examples.babi.run import nlu_model_path
from examples.restaurant_example import RestaurantPolicy
from rasa_core.agent import Agent
from rasa_core.channels.file import FileInputChannel
from rasa_core.interpreter import RegexInterpreter, RasaNLUInterpreter
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


def run_babi_online():
    training_data = 'examples/babi/data/babi_task5_dev_rasa_even_smaller.md'
    logger.info("Starting to train policy")
    agent = Agent("examples/restaurant_domain.yml",
                  policies=[MemoizationPolicy(), RestaurantPolicy()],
                  interpreter=RegexInterpreter())

    input_c = FileInputChannel(training_data,
                               message_line_pattern='^\s*\*\s(.*)$',
                               max_messages=10)
    agent.train_online(training_data,
                       input_channel=input_c,
                       epochs=10)

    agent.interpreter = RasaNLUInterpreter(nlu_model_path)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_babi_online()
