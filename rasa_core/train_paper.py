from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import pickle
import os

from rasa_core.agent import Agent
from rasa_core.featurizers import (LabelTokenizerSingleStateFeaturizer,
                                   FullDialogueTrackerFeaturizer,
                                   MaxHistoryTrackerFeaturizer)
from rasa_core.policies.embedding_policy import EmbeddingPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.domain import TemplateDomain
from rasa_core.training.dsl import StoryFileReader
from rasa_core import utils

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for the evaluate script."""

    parser = argparse.ArgumentParser(
            description='Trains multiple Core models with the Embedding and '
                        'Keras policies')
    parser.add_argument(
            '--exclude',
            type=str,
            required=True,
            help="file to exclude from training")
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing the training stories")
    parser.add_argument(
            '--epochs_embed',
            type=int,
            default=2000,
            help="number of epochs for the embedding policy")
    parser.add_argument(
            '--epochs_keras',
            type=int,
            default=400,
            help="number of epochs for the keras policy")
    parser.add_argument(
            '-o', '--out',
            type=str,
            default='models/',
            help="directory to persist the trained model in")
    parser.add_argument(
            '--percentages',
            nargs="*",
            type=int,
            default=[0, 5, 25, 50, 70, 90, 95, 100],
            help="Range of exclusion percentages")
    parser.add_argument(
            '--runs',
            type=int,
            default=3,
            help="Number of runs for experiments")
    parser.add_argument(
            '-d', '--domain',
            type=str,
            required=True,
            help="domain specification yaml file")

    utils.add_logging_option_arguments(parser)
    return parser


def train_domain_policy(story_filename,
                        domain,
                        epochs,
                        output_path=None,
                        exclusion_file=None,
                        exclusion_percentage=None,
                        starspace=True):

    """Trains either a KerasPolicy model or an EmbeddingPolicy, excluding a
    certain percentage of a story file"""

    if starspace:
        featurizer = FullDialogueTrackerFeaturizer(
                        LabelTokenizerSingleStateFeaturizer())
        policies = [EmbeddingPolicy(featurizer)]
    else:
        featurizer = MaxHistoryTrackerFeaturizer(
                        LabelTokenizerSingleStateFeaturizer(),
                        max_history=20)
        policies = [KerasPolicy(featurizer)]

    agent = Agent(domain,
                  policies=policies)

    data = agent.load_data(story_filename,
                           remove_duplicates=True,
                           augmentation_factor=0,
                           exclusion_file=exclusion_file,
                           exclusion_percentage=exclusion_percentage)

    agent.train(data,
                rnn_size=64,
                epochs=epochs,
                embed_dim=20,
                attn_shift_range=5)

    agent.persist(model_path=output_path)


def get_no_of_stories(file, domain):

    """gets number of stories in a file"""

    no_stories = len(StoryFileReader.read_from_file(file,
                                                    TemplateDomain.load(
                                                        domain)))
    return no_stories


if __name__ == '__main__':
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    for r in range(cmdline_args.runs):
        logging.info("Starting run {}/{}".format(r + 1, cmdline_args.runs))
        for i in cmdline_args.percentages:

            current_round = cmdline_args.percentages.index(i) + 1

            output_path_keras = (os.path.join(cmdline_args.out, 'run_' +
                                 str(r + 1), 'keras' + str(current_round)))

            output_path_embed = (os.path.join(cmdline_args.out, 'run_' +
                                 str(r + 1), 'embed' + str(current_round)))

            logging.info("Starting to train embed round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

            train_domain_policy(story_filename=cmdline_args.stories,
                                domain=cmdline_args.domain,
                                epochs=cmdline_args.epochs_embed,
                                output_path=output_path_embed,
                                exclusion_file=cmdline_args.exclude,
                                exclusion_percentage=i,
                                starspace=True)

            logger.info("Finished training embed round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

            logging.info("Starting to train keras round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

            train_domain_policy(story_filename=cmdline_args.stories,
                                domain=cmdline_args.domain,
                                epochs=cmdline_args.epochs_keras,
                                output_path=output_path_keras,
                                exclusion_file=cmdline_args.exclude,
                                exclusion_percentage=i,
                                starspace=False)

            logger.info("Finished training keras round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

    no_stories = get_no_of_stories(cmdline_args.exclude, cmdline_args.domain)

    # store the list of the number of stories excluded at each exclusion
    # percentage
    story_range = [no_stories - round((x/100.0) * no_stories) for x in
                   cmdline_args.percentages]

    pickle.dump(story_range,
                open(os.path.join(cmdline_args.out, 'num_stories.p'), 'wb'))
