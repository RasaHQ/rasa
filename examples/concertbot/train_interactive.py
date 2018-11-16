import logging

from rasa_core import utils, train
from rasa_core.training import interactive

logger = logging.getLogger(__name__)


def train_agent():
    return train.train_dialogue_model(domain_file="domain.yml",
                                      stories_file="data/stories.md",
                                      output_path="models/dialogue",
                                      kwargs={"batch_size": 50,
                                              "epochs": 200,
                                              "max_training_samples": 300
                                              })


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")
    agent = train_agent()
    interactive.run_interactive_learning(agent)
