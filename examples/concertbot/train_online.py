from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.channels import online
from rasa_core.domain import TemplateDomain

logger = logging.getLogger(__name__)


def run_concertbot_online(interpreter,
                          domain_file="concert_domain.yml",
                          training_data_file='data/stories.md',
                          endpoints="endpoints.yml"):
    action_endpoint = utils.read_endpoint_config(endpoints, "action_endpoint")
    domain = TemplateDomain.load(domain_file,
                                 action_endpoint)
    agent = Agent(domain,
                  policies=[MemoizationPolicy(max_history=2), KerasPolicy()],
                  interpreter=interpreter)

    training_data = agent.load_data(training_data_file)
    agent.train(training_data,
                batch_size=50,
                epochs=200,
                max_training_samples=300)
    online.serve_application(agent)

    return agent


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")
    run_concertbot_online(RegexInterpreter())
