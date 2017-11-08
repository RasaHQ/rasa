from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging

from builtins import str

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RasaNLUInterpreter, RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy


def create_argument_parser():
    parser = argparse.ArgumentParser(
            description='trains a dialogue model')
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file that contains the stories to train on")
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
            '-v', '--verbose',
            default=True,
            help="use verbose logging")
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

    return parser


def train_dialogue_model(domain_file, stories_file, output_path,
                         use_online_learning=False, nlu_model_path=None,
                         kwargs={}):
    agent = Agent(domain_file, policies=[MemoizationPolicy(), KerasPolicy()])

    if use_online_learning:
        if nlu_model_path:
            agent.interpreter = RasaNLUInterpreter(nlu_model_path)
        else:
            agent.interpreter = RegexInterpreter()
        agent.train_online(
                stories_file,
                input_channel=ConsoleInputChannel(),
                epochs=10,
                model_path=output_path)
    else:
        agent.train(
                stories_file,
                validation_split=0.1,
                **kwargs
        )

    agent.persist(output_path)


if __name__ == '__main__':

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.basicConfig(level="DEBUG" if cmdline_args.verbose else "INFO")

    additional_arguments = {
        "max_history": cmdline_args.history,
        "epochs": cmdline_args.epochs,
        "batch_size": cmdline_args.batch_size,
        "augmentation_factor": cmdline_args.augmentation
    }

    train_dialogue_model(cmdline_args.domain,
                         cmdline_args.stories,
                         cmdline_args.out,
                         cmdline_args.online,
                         cmdline_args.nlu,
                         additional_arguments)
