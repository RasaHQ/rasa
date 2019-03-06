import argparse
import logging
import os
import tempfile
from typing import Text, Dict, Optional

import rasa_core.cli.train
from rasa_core import config, cli
from rasa_core import utils
from rasa_core.broker import PikaProducer
from rasa_core.domain import TemplateDomain
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.run import AvailableEndpoints
from rasa_core.tracker_store import TrackerStore
from rasa_core.training.dsl import StoryFileReader
from rasa_core.utils import set_default_subparser

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        description='Train a dialogue model for Rasa Core. '
                    'The training will use your conversations '
                    'in the story training data format and '
                    'your domain definition to train a dialogue '
                    'model to predict a bots actions.')
    parent_parser = argparse.ArgumentParser(add_help=False)
    cli.train.add_general_args(parent_parser)

    subparsers = parser.add_subparsers(
        help='Training mode of core.',
        dest='mode')
    subparsers.required = True

    train_parser = subparsers.add_parser(
        'default',
        help='train a dialogue model',
        parents=[parent_parser])
    compare_parser = subparsers.add_parser(
        'compare',
        help='train multiple dialogue models to compare '
             'policies',
        parents=[parent_parser])
    interactive_parser = subparsers.add_parser(
        'interactive',
        help='teach the bot with interactive learning',
        parents=[parent_parser])

    cli.train.add_compare_args(compare_parser)
    cli.train.add_interactive_args(interactive_parser)
    cli.train.add_train_args(train_parser)

    return parser


def train(domain_file: Text, stories_file: Text, output_path: Text,
          interpreter: Optional[NaturalLanguageInterpreter] = None,
          endpoints: AvailableEndpoints = AvailableEndpoints(),
          dump_stories: bool = False,
          policy_config: Text = None,
          exclusion_percentage: int = None,
          kwargs: Optional[Dict] = None):
    from rasa_core.agent import Agent

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
        "augmentation_factor": args.augmentation,
        "debug_plots": args.debug_plots
    }
    # remove None values
    return {k: v for k, v in additional.items() if v is not None}


def train_comparison_models(stories,
                            domain,
                            output_path="",
                            exclusion_percentages=None,
                            policy_configs=None,
                            runs=1,
                            dump_stories=False,
                            kwargs=None):
    """Train multiple models for comparison of policies"""

    exclusion_percentages = exclusion_percentages or []
    policy_configs = policy_configs or []

    for r in range(runs):
        logging.info("Starting run {}/{}".format(r + 1, runs))

        for i in exclusion_percentages:
            current_round = exclusion_percentages.index(i) + 1

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
                             " with {}% exclusion"
                             "".format(policy_name, current_round,
                                       len(exclusion_percentages), i))

                train(
                    domain, stories, output,
                    policy_config=policy_config,
                    exclusion_percentage=i,
                    kwargs=kwargs,
                    dump_stories=dump_stories)


def get_no_of_stories(story_file, domain):
    """Get number of stories in a file."""

    stories = StoryFileReader.read_from_folder(story_file,
                                               TemplateDomain.load(domain))
    return len(stories)


def do_default_training(cmdline_args, stories, additional_arguments):
    """Train a model."""

    train(domain_file=cmdline_args.domain,
          stories_file=stories,
          output_path=cmdline_args.out,
          dump_stories=cmdline_args.dump_stories,
          policy_config=cmdline_args.config[0],
          kwargs=additional_arguments)


def do_compare_training(cmdline_args, stories, additional_arguments):
    train_comparison_models(stories,
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
    story_range = [no_stories - round((x / 100.0) * no_stories)
                   for x in cmdline_args.percentages]

    story_n_path = os.path.join(cmdline_args.out, 'num_stories.json')
    utils.dump_obj_as_json_to_file(story_n_path, story_range)


def do_interactive_learning(cmdline_args, stories, additional_arguments=None):
    _endpoints = AvailableEndpoints.read_endpoints(cmdline_args.endpoints)
    _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                     _endpoints.nlu)
    from rasa_core.agent import Agent
    from rasa_core.training import interactive

    if cmdline_args.core:
        if cmdline_args.finetune:
            raise ValueError("--core can only be used without "
                             "--finetune flag.")

        logger.info("Loading a pre-trained model. This means that "
                    "all training-related parameters will be ignored.")

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
        if cmdline_args.out:
            model_directory = cmdline_args.out
        else:
            model_directory = tempfile.mkdtemp(suffix="_core_model")

        _agent = train(cmdline_args.domain,
                       stories,
                       model_directory,
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


if __name__ == '__main__':

    # Running as standalone python application
    arg_parser = create_argument_parser()
    set_default_subparser(arg_parser, 'default')
    cmdline_arguments = arg_parser.parse_args()
    additional_args = _additional_arguments(cmdline_arguments)

    utils.configure_colored_logging(cmdline_arguments.loglevel)

    training_stories = rasa_core.cli.train.stories_from_cli_args(
        cmdline_arguments)

    if cmdline_arguments.mode == 'default':
        do_default_training(cmdline_arguments,
                            training_stories,
                            additional_args)

    elif cmdline_arguments.mode == 'interactive':
        do_interactive_learning(cmdline_arguments,
                                training_stories,
                                additional_args)

    elif cmdline_arguments.mode == 'compare':
        do_compare_training(cmdline_arguments,
                            training_stories,
                            additional_args)
