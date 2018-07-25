from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

from builtins import str

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter, RegexInterpreter
from rasa_core.featurizers import \
    MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy


def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
            description='trains a dialogue model')
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing the training stories")
    parser.add_argument(
            '-o', '--out',
            type=str,
            required=True,
            help="directory to persist the trained model in")
    parser.add_argument(
            '-d', '--domain',
            type=str,
            required=True,
            help="domain specification yaml file")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            default=None,
            help="trained nlu model")
    parser.add_argument(
            '--history',
            type=int,
            default=3,
            help="max history to use of a story")
    parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help="number of epochs to train the model")
    parser.add_argument(
            '--validation_split',
            type=float,
            default=0.1,
            help="Percentage of training samples used for validation, "
                 "0.1 by default")
    parser.add_argument(
            '--batch_size',
            type=int,
            default=20,
            help="number of training samples to put into one training batch")
    parser.add_argument(
            '--online',
            default=False,
            action='store_true',
            help="enable online training")
    parser.add_argument(
            '--augmentation',
            type=int,
            default=50,
            help="how much data augmentation to use during training")
    parser.add_argument(
            '--debug_plots',
            default=False,
            action='store_true',
            help="If enabled, will create plots showing checkpoints "
                 "and their connections between story blocks in a  "
                 "file called `story_blocks_connections.pdf`.")
    parser.add_argument(
            '--dump_stories',
            default=False,
            action='store_true',
            help="If enabled, save flattened stories to a file")

    utils.add_logging_option_arguments(parser)
    return parser


def train_dialogue_model(domain_file, stories_file, output_path,
                         use_online_learning=False,
                         nlu_model_path=None,
                         max_history=None,
                         dump_flattened_stories=False,
                         kwargs=None):
    if not kwargs:
        kwargs = {}

    agent = Agent(domain_file, policies=[
        MemoizationPolicy(max_history=max_history),
        KerasPolicy(MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                                max_history=max_history))])

    data_load_args, kwargs = utils.extract_args(kwargs,
                                                {"use_story_concatenation",
                                                 "unique_last_num_states",
                                                 "augmentation_factor",
                                                 "remove_duplicates",
                                                 "debug_plots"})
    training_data = agent.load_data(stories_file, **data_load_args)

    if use_online_learning:
        if nlu_model_path:
            agent.interpreter = RasaNLUInterpreter(nlu_model_path)
        else:
            agent.interpreter = RegexInterpreter()
        agent.train_online(
                training_data,
                input_channel=ConsoleInputChannel(),
                model_path=output_path,
                **kwargs)
    else:
        agent.train(training_data, **kwargs)

    agent.persist(output_path, dump_flattened_stories)


if __name__ == '__main__':

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    additional_arguments = {
        "epochs": cmdline_args.epochs,
        "batch_size": cmdline_args.batch_size,
        "validation_split": cmdline_args.validation_split,
        "augmentation_factor": cmdline_args.augmentation,
        "debug_plots": cmdline_args.debug_plots
    }

    train_dialogue_model(cmdline_args.domain,
                         cmdline_args.stories,
                         cmdline_args.out,
                         cmdline_args.online,
                         cmdline_args.nlu,
                         cmdline_args.history,
                         cmdline_args.dump_stories,
                         additional_arguments)
