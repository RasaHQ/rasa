import argparse
import logging
import os

import rasa_core.cli.arguments
from rasa_core import utils, config, cli
from rasa_core.agent import Agent

logger = logging.getLogger(__name__)


def add_arguments(parser):
    """Parse all the command line arguments for the visualisation script."""
    rasa_core.cli.arguments.add_logging_option_arguments(parser)
    cli.visualization.add_visualization_arguments(parser)
    cli.arguments.add_config_arg(parser, nargs=1)
    cli.arguments.add_domain_arg(parser)
    cli.arguments.add_model_and_story_group(parser,
                                            allow_pretrained_model=False)
    return parser


def visualize(args):
    utils.configure_colored_logging(args.loglevel)

    policies = config.load(args.config[0])

    agent = Agent(args.domain, policies=policies)

    # this is optional, only needed if the `/greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if args.nlu_data is not None:
        from rasa_nlu.training_data import load_data

        nlu_data = load_data(args.nlu_data)
    else:
        nlu_data = None

    stories = cli.stories_from_cli_args(args)

    logger.info("Starting to visualize stories...")
    agent.visualize(stories, args.output,
                    args.max_history,
                    nlu_training_data=nlu_data)

    full_output_path = "file://{}".format(os.path.abspath(args.output))
    logger.info("Finished graph creation. Saved into file://{}".format(
        full_output_path))

    import webbrowser
    webbrowser.open("file://{}".format(full_output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the stories in a dialogue training file')

    arg_parser = add_arguments(parser)
    cmdline_arguments = arg_parser.parse_args()

    visualize(cmdline_arguments)


