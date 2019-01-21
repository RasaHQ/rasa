import argparse
import logging
import os

from rasa_core import utils, config, cli
from rasa_core.agent import Agent

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the visualisation script."""

    parser = argparse.ArgumentParser(
        description='Visualize the stories in a dialogue training file')

    parser.add_argument(
        '-o', '--output',
        required=True,
        type=str,
        help="filename of the output path, e.g. 'graph.html")
    parser.add_argument(
        '-m', '--max_history',
        default=2,
        type=int,
        help="max history to consider when merging "
             "paths in the output graph")
    parser.add_argument(
        '-nlu', '--nlu_data',
        default=None,
        type=str,
        help="path of the Rasa NLU training data, "
             "used to insert example messages into the graph")

    utils.add_logging_option_arguments(parser)

    cli.arguments.add_config_arg(parser, nargs=1)
    cli.arguments.add_domain_arg(parser)
    cli.arguments.add_model_and_story_group(parser,
                                            allow_pretrained_model=False)
    return parser


if __name__ == '__main__':
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_arguments.loglevel)

    policies = config.load(cmdline_arguments.config[0])

    agent = Agent(cmdline_arguments.domain, policies=policies)

    # this is optional, only needed if the `/greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if cmdline_arguments.nlu_data is not None:
        from rasa_nlu.training_data import load_data

        nlu_data = load_data(cmdline_arguments.nlu_data)
    else:
        nlu_data = None

    stories = cli.stories_from_cli_args(cmdline_arguments)

    logger.info("Starting to visualize stories...")
    agent.visualize(stories, cmdline_arguments.output,
                    cmdline_arguments.max_history,
                    nlu_training_data=nlu_data)

    logger.info("Finished graph creation. Saved into file://{}".format(
        os.path.abspath(cmdline_arguments.output)))
