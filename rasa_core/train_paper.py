from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import pickle

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
            description='evaluates a dialogue model')
    parser.add_argument(
            '--exclude',
            type=str,
            required=True,
            help="file to exclude from training")
    parser.add_argument(
            '--data',
            type=str,
            required=True,
            help="Data to train on")
    parser.add_argument(
            '--epochs',
            type=int,
            default=2000,
            help="number of epochs")
    parser.add_argument(
            '--path',
            type=str,
            default='models/',
            help="Path to store models in")
    parser.add_argument(
            '--percentages',
            type=list,
            default=[0, 5, 25, 50, 70, 90, 95, 100],
            help="Range of exclusion percentages")
    parser.add_argument(
            '--runs',
            type=int,
            default=3,
            help="Number of runs for experiments")
    parser.add_argument(
            '--domain',
            type=str,
            required=True,
            help="Path of domain file")

    utils.add_logging_option_arguments(parser)
    return parser


def train_domain_policy(story_filename,
                        domain,
                        output_path=None,
                        exclusion_file=None,
                        exclusion_percentage=None,
                        starspace=True,
                        epoch_no=2000):
    """Trains a new deterministic domain policy using the stories
    (json format) in `story_filename`."""
    if starspace:
        featurizer = FullDialogueTrackerFeaturizer(
                        LabelTokenizerSingleStateFeaturizer())
        policies = [EmbeddingPolicy(featurizer)]
        epochs = epoch_no
        batch_size = [8, 32]
    else:
        featurizer = MaxHistoryTrackerFeaturizer(
                        LabelTokenizerSingleStateFeaturizer(),
                        max_history=20)
        policies = [KerasPolicy(featurizer)]
        epochs = 400
        batch_size = 32

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
                use_attention=True,
                skip_cells=True,
                attn_shift_range=5,
                batch_size=batch_size)

    agent.persist(model_path=output_path)


def get_no_of_stories(exclude, domain):
    no_stories = len(StoryFileReader.read_from_file(exclude,
                                                    TemplateDomain.load(
                                                        domain)))
    return no_stories


if __name__ == '__main__':
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)
    for r in xrange(1, cmdline_args.runs):
        for i in cmdline_args.percentages:
            current_round = cmdline_args.percentages.index(i) + 1
            output_path_keras = (cmdline_args.path + 'run_' + str(r) + '/keras'
                                 + str(current_round))
            output_path_embed = (cmdline_args.path + 'run_' + str(r) + '/embed'
                                 + str(current_round))
            logging.info("Starting to train embed round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

            train_domain_policy(story_filename=cmdline_args.data,
                                domain=cmdline_args.domain,
                                output_path=output_path_embed,
                                exclusion_file=cmdline_args.exclude,
                                exclusion_percentage=i,
                                starspace=True,
                                epoch_no=cmdline_args.epochs)
            logger.info("Finished training embed round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

            logging.info("Starting to train keras round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))

            train_domain_policy(story_filename=cmdline_args.data,
                                domain=cmdline_args.domain,
                                output_path=output_path_keras,
                                exclusion_file=cmdline_args.exclude,
                                exclusion_percentage=i,
                                starspace=False)

            logger.info("Finished training keras round {}/{}".format(
                                                current_round,
                                                len(cmdline_args.percentages)))
    no_stories = get_no_of_stories(cmdline_args.exclude, cmdline_args.domain)
    story_range = [no_stories - round((x/100.0) * no_stories) for x in
                   cmdline_args.percentages]
    pickle.dump(story_range, open(cmdline_args.path + 'num_stories.p', 'wb'))
