import logging

from rasa_core import utils, train
from rasa_core.training import interactive

logger = logging.getLogger(__name__)


def train_agent():
    return train(domain_file="domain.yml",
                 stories_file="data/stories.md",
                 output_path="models/dialogue",
                 policy_config='policy_config.yml')


if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")
    agent = train_agent()
    logger.info("This example does not include NLU data."
                "Please specify the desired intent with a preceding '/', e.g."
                "'/greet' .")
    interactive.run_interactive_learning(agent)
