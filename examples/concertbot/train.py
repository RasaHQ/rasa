from rasa_core import train, utils

import logging
logger = logging.getLogger(__name__)


def train_dialogue(domain_file='domain.yml',
                   stories_file='data/stories.md',
                   model_path='models/dialogue',
                   policy_config='policy_config.yml'):
    return train(domain_file=domain_file,
                 stories_file=stories_file,
                 output_path=model_path,
                 policy_config=policy_config,
                 kwargs={'augmentation_factor': 50, 'validation_split': 0.2})


if __name__ == '__main__':
    train_dialogue()

    utils.configure_colored_logging(loglevel="INFO")
    logger.info("This example does not include NLU data."
                "Please specify the desired intent with a preceding '/', e.g."
                "'/greet' .")
