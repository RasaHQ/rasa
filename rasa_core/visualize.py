import argparse
import logging
import os
from typing import Text

import rasa_core.cli.arguments
import rasa_core.cli.train
from rasa_core import utils
import rasa_core.cli

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser):
    """Parse all the command line arguments for the visualisation script."""
    rasa_core.cli.arguments.add_logging_option_arguments(parser)
    rasa_core.cli.visualization.add_visualization_arguments(parser)
    rasa_core.cli.arguments.add_config_arg(parser, nargs=1)
    rasa_core.cli.arguments.add_domain_arg(parser)
    rasa_core.cli.arguments.add_model_and_story_group(
        parser, allow_pretrained_model=False)
    return parser


def visualize(config_path: Text, domain_path: Text, stories_path: Text,
              nlu_data_path: Text, output_path: Text, max_history: int):
    from rasa_core.agent import Agent
    from rasa_core import config

    policies = config.load(config_path)

    agent = Agent(domain_path, policies=policies)

    # this is optional, only needed if the `/greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if nlu_data_path is not None:
        from rasa_nlu.training_data import load_data

        nlu_data_path = load_data(nlu_data_path)
    else:
        nlu_data_path = None

    logger.info("Starting to visualize stories...")
    agent.visualize(stories_path, output_path,
                    max_history,
                    nlu_training_data=nlu_data_path)

    full_output_path = "file://{}".format(os.path.abspath(output_path))
    logger.info("Finished graph creation. Saved into {}".format(
        full_output_path))

    import webbrowser
    webbrowser.open(full_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the stories in a dialogue training file')
    arg_parser = add_arguments(parser)
    args = arg_parser.parse_args()

    utils.configure_colored_logging(args.loglevel)
    stories = rasa_core.cli.train.stories_from_cli_args(args)

    visualize(args.config[0], args.domain, stories, args.nlu_data,
              args.output, args.max_history)
