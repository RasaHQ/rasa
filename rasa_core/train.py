from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str

import argparse
import logging
import io
import os
import pickle

from rasa_core import config
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.domain import TemplateDomain
from rasa_core.broker import PikaProducer
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.run import AvailableEndpoints
from rasa_core.training import interactive
from rasa_core.tracker_store import TrackerStore
from rasa_core.training.dsl import StoryFileReader
logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
            description='trains a dialogue model')
    parent_parser = argparse.ArgumentParser(add_help=False)
    add_args_to_parser(parent_parser)
    add_model_and_story_group(parent_parser)
    utils.add_logging_option_arguments(parent_parser)
    subparsers = parser.add_subparsers(help='mode', dest='mode')
    subparsers.add_parser('default',
                          help='default mode: train a dialogue'
                               ' model',
                               parents=[parent_parser])
    compare_parser = subparsers.add_parser('compare',
                                           help='compare mode: train multiple '
                                                'dialogue models to compare '
                                                'policies',
                                           parents=[parent_parser])
    interactive_parser = subparsers.add_parser('interactive',
                                               help='teach the bot with '
                                                    'interactive learning',
                                               parents=[parent_parser])
    add_compare_args(compare_parser)
    add_interactive_args(interactive_parser)

    return parser


def add_compare_args(parser):
    parser.add_argument(
            '--percentages',
            nargs="*",
            type=int,
            default=[0, 5, 25, 50, 70, 90, 95],
            help="Range of exclusion percentages")
    parser.add_argument(
            '--runs',
            type=int,
            default=3,
            help="Number of runs for experiments")


def add_interactive_args(parser):
        parser.add_argument(
                '-u', '--nlu',
                type=str,
                default=None,
                help="trained nlu model")
        parser.add_argument(
                '--endpoints',
                default=None,
                help="Configuration file for the connectors as a yml file")
        parser.add_argument(
                '--skip_visualization',
                default=False,
                action='store_true',
                help="disables plotting the visualization during "
                     "interactive learning")
        parser.add_argument(
                '--finetune',
                default=False,
                action='store_true',
                help="retrain the model immediately based on feedback.")


def add_args_to_parser(parser):

    parser.add_argument(
            '-o', '--out',
            type=str,
            required=False,
            help="directory to persist the trained model in")
    parser.add_argument(
            '-d', '--domain',
            type=str,
            required=True,
            help="domain specification yaml file")
    parser.add_argument(
            '--validation_split',
            type=float,
            default=0.1,
            help="Percentage of training samples used for validation, "
                 "0.1 by default")
    parser.add_argument(
            '--augmentation',
            type=int,
            default=50,
            help="how much data augmentation to use during training")
    parser.add_argument(
            '-c', '--config',
            type=str,
            nargs="*",
            required=False,
            help="Policy specification yaml file.")
    parser.add_argument(
            '--dump_stories',
            default=False,
            action='store_true',
            help="If enabled, save flattened stories to a file")
    parser.add_argument(
            '--debug_plots',
            default=False,
            action='store_true',
            help="If enabled, will create plots showing checkpoints "
                 "and their connections between story blocks in a  "
                 "file called `story_blocks_connections.pdf`.")

    return parser


def add_model_and_story_group(parser):
    # either the user can pass in a story file, or the data will get
    # downloaded from a url
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
            '-s', '--stories',
            type=str,
            help="file or folder containing the training stories")
    group.add_argument(
            '--url',
            type=str,
            help="If supplied, downloads a story file from a URL and "
                 "trains on it. Fetches the data by sending a GET request "
                 "to the supplied URL.")
    group.add_argument(
            '--core',
            default=None,
            help="path to load a pre-trained model instead of training (for "
                 "interactive mode only)")
    return parser


def train_dialogue_model(domain_file, stories_file, output_path,
                         interpreter=None,
                         endpoints=AvailableEndpoints(),
                         dump_stories=False,
                         policy_config=None,
                         exclusion_percentage=None,
                         kwargs=None):
    if not kwargs:
        kwargs = {}

    policies = config.load(policy_config)

    agent = Agent(domain_file,
                  generator=endpoints.nlg,
                  action_endpoint=endpoints.action,
                  interpreter=interpreter,
                  policies=policies)

    data_load_args, kwargs = utils.extract_args(kwargs,
                                                {"use_story_concatenation",
                                                 "unique_last_num_states",
                                                 "augmentation_factor",
                                                 "remove_duplicates",
                                                 "debug_plots"})

    training_data = agent.load_data(stories_file,
                                    exclusion_percentage=exclusion_percentage,
                                    **data_load_args)
    agent.train(training_data, **kwargs)
    agent.persist(output_path, dump_stories)

    return agent


def _additional_arguments(args):
    additional = {
        "validation_split": args.validation_split,
        "augmentation_factor": args.augmentation,
        "debug_plots": args.debug_plots
    }

    # remove None values
    return {k: v for k, v in additional.items() if v is not None}


def train_comparison_models(story_filename,
                            domain,
                            output_path=None,
                            exclusion_percentages=None,
                            policy_configs=None,
                            runs=None,
                            dump_stories=False,
                            kwargs=None):

    """Trains either a KerasPolicy model or an EmbeddingPolicy, excluding a
    certain percentage of a story file"""
    for r in range(cmdline_args.runs):
        logging.info("Starting run {}/{}".format(r + 1, cmdline_args.runs))
        for i in exclusion_percentages:
            current_round = cmdline_args.percentages.index(i) + 1
            for policy_config in policy_configs:
                policies = config.load(policy_config)
                if len(policies) > 1:
                    raise ValueError("You can only specify one policy per "
                                     "model for comparison")
                policy_name = type(policies[0]).__name__
                output = os.path.join(output_path, 'run_' + str(r + 1),
                                      policy_name +
                                      str(current_round))

                logging.info("Starting to train {} round {}/{}"
                             " with {}% exclusion".format(
                                                policy_name,
                                                current_round,
                                                len(exclusion_percentages),
                                                i))

                train_dialogue_model(
                            domain, stories, output,
                            policy_config=policy_config,
                            exclusion_percentage=i,
                            kwargs=kwargs,
                            dump_stories=dump_stories)


def get_no_of_stories(stories, domain):

    """gets number of stories in a file"""

    no_stories = len(StoryFileReader.read_from_folder(stories,
                                                      TemplateDomain.load(
                                                            domain)))
    return no_stories


if __name__ == '__main__':

    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    additional_arguments = _additional_arguments(cmdline_args)

    utils.configure_colored_logging(cmdline_args.loglevel)

    if cmdline_args.url:
        stories = utils.download_file_from_url(cmdline_args.url)
    else:
        stories = cmdline_args.stories

    if cmdline_args.mode == 'default':
        if not cmdline_args.out:
            raise ValueError("you must provide a path where the model "
                             "will be saved using -o / --out")
        if (isinstance(cmdline_args.config, list) and
                len(cmdline_args.config) > 1):
            raise ValueError("You can only pass one config file at a time")

        _agent = train_dialogue_model(domain_file=cmdline_args.domain,
                                      stories_file=stories,
                                      output_path=cmdline_args.out,
                                      dump_stories=cmdline_args.dump_stories,
                                      policy_config=cmdline_args.config[0],
                                      kwargs=additional_arguments)

    elif cmdline_args.mode == 'interactive':
        _endpoints = AvailableEndpoints.read_endpoints(cmdline_args.endpoints)
        _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                         _endpoints.nlu)
        if (isinstance(cmdline_args.config, list) and
                len(cmdline_args.config) > 1):
            raise ValueError("You can only pass one config file at a time")
        if cmdline_args.core and cmdline_args.finetune:
            raise ValueError("--core can only be used without --finetune flag.")
        elif cmdline_args.core:
            logger.info("loading a pre-trained model. "
                        "all training-related parameters will be ignored")

            _broker = PikaProducer.from_endpoint_config(_endpoints.event_broker)
            _tracker_store = TrackerStore.find_tracker_store(
                                                None,
                                                _endpoints.tracker_store,
                                                _broker)
            _agent = Agent.load(cmdline_args.core,
                                interpreter=_interpreter,
                                generator=_endpoints.nlg,
                                tracker_store=_tracker_store,
                                action_endpoint=_endpoints.action)
        else:
            if not cmdline_args.out:
                raise ValueError("you must provide a path where the model "
                                 "will be saved using -o / --out")

            _agent = train_dialogue_model(cmdline_args.domain,
                                          stories,
                                          cmdline_args.out,
                                          _interpreter,
                                          _endpoints,
                                          cmdline_args.dump_stories,
                                          cmdline_args.config[0],
                                          None,
                                          additional_arguments)
        interactive.run_interactive_learning(
                _agent, stories,
                finetune=cmdline_args.finetune,
                skip_visualization=cmdline_args.skip_visualization)

    elif cmdline_args.mode == 'compare':
        if not cmdline_args.out:
            raise ValueError("you must provide a path where the model "
                             "will be saved using -o / --out")

        train_comparison_models(cmdline_args.stories,
                                cmdline_args.domain,
                                cmdline_args.out,
                                cmdline_args.percentages,
                                cmdline_args.config,
                                cmdline_args.runs,
                                cmdline_args.dump_stories,
                                additional_arguments)

        no_stories = get_no_of_stories(cmdline_args.stories,
                                       cmdline_args.domain)

        # store the list of the number of stories present at each exclusion
        # percentage
        story_range = [no_stories - round((x/100.0) * no_stories) for x in
                       cmdline_args.percentages]

        pickle.dump(story_range,
                    io.open(os.path.join(cmdline_args.out, 'num_stories.p'),
                            'wb'))
