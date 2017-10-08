from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

import os

from rasa_core.domain import TemplateDomain
from rasa_core.training_utils.dsl import StoryFileReader
from rasa_core.training_utils.visualization import visualize_stories

logger = logging.getLogger(__name__)


def create_argument_parser():
    parser = argparse.ArgumentParser(
            description='Visualize the stories in a dialogue training file')

    parser.add_argument('-s', '--stories',
                        required=True,
                        help="story file")
    parser.add_argument('-d', '--domain',
                        required=True,
                        help="domain file")
    parser.add_argument('-o', '--output',
                        required=True,
                        help="filename of the output path, e.g. 'graph.png")
    parser.add_argument('-m', '--max_history',
                        default=2,
                        type=int,
                        help="max history to consider when merging "
                             "paths in the output graph")
    parser.add_argument('-nlu', '--nlu_data',
                        default=None,
                        help="path of the Rasa NLU training data, "
                             "used to insert example messages into the graph")
    return parser


if __name__ == '__main__':
    parser = create_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level="DEBUG")

    domain = TemplateDomain.load(args.domain)
    story_steps = StoryFileReader.read_from_file(args.stories, domain)

    # this is optional, only needed if the `_greet` type of
    # messages in the stories should be replaced with actual
    # messages (e.g. `hello`)
    if args.nlu_data is not None:
        from rasa_nlu.converters import load_data

        nlu_data = load_data(args.nlu_data)
    else:
        nlu_data = None

    logger.info("Starting to visualize stories...")
    visualize_stories(story_steps, args.output, args.max_history,
                      training_data=nlu_data)

    logger.info("Finished graph creation. Saved into {}".format(
            os.path.abspath(args.output)))
