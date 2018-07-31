from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core import utils, train
from rasa_core.training import online

logger = logging.getLogger(__name__)


def train_agent():
    return train.train_dialogue_model(domain_file="domain.yml",
                                      stories_file="data/stories.md",
                                      output_path="models/dialogue",
                                      endpoints="endpoints.yml",
                                      max_history=2,
                                      kwargs={"batch_size": 50,
                                              "epochs": 200,
                                              "max_training_samples": 300
                                              })


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")
    agent = train_agent()
    online.serve_agent(agent)
