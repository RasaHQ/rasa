from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os

from builtins import str

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the visualisation script."""

    parser = argparse.ArgumentParser(
            description='Visualize the stories in a dialogue training file')

    parser.add_argument('-s', '--stories',
                        required=True,
                        type=str,
                        help="story file or folder containing the stories"
                             "to visualize")
    parser.add_argument('-d', '--domain',
                        required=True,
                        type=str,
                        help="domain file")
    parser.add_argument('-o', '--output',
                        required=True,
                        type=str,
                        help="filename of the output path, e.g. 'graph.png")
    parser.add_argument('-m', '--max_history',
                        default=2,
                        type=int,
                        help="max history to consider when merging "
                             "paths in the output graph")
    parser.add_argument('-nlu', '--nlu_data',
                        default=None,
                        type=str,
                        help="path of the Rasa NLU training data, "
                             "used to insert example messages into the graph")

    utils.add_logging_option_arguments(parser)
    return parser


if __name__ == '__main__':
    arg_parser = create_argument_parser()
    args = arg_parser.parse_args()

    utils.configure_colored_logging(args.loglevel)

    agent = Agent(args.domain, policies=[MemoizationPolicy(), KerasPolicy()])

    # this is optional, only needed if the `_greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if args.nlu_data is not None:
        from rasa_nlu.training_data import load_data

        nlu_data = load_data(args.nlu_data)
    else:
        nlu_data = None

    logger.info("Starting to visualize stories...")
    agent.visualize(args.stories, args.output, args.max_history,
                    nlu_training_data=nlu_data)

    logger.info("Finished graph creation. Saved into {}".format(
            os.path.abspath(args.output)))
