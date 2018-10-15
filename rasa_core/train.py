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

from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.constants import (
    DEFAULT_NLU_FALLBACK_THRESHOLD,
    DEFAULT_CORE_FALLBACK_THRESHOLD, DEFAULT_FALLBACK_ACTION)
from rasa_core.domain import TemplateDomain
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer, BinarySingleStateFeaturizer,
    FullDialogueTrackerFeaturizer, LabelTokenizerSingleStateFeaturizer)
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.policies import FallbackPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.embedding_policy import EmbeddingPolicy
from rasa_core.run import AvailableEndpoints
from rasa_core.training import interactive
from rasa_core.training.dsl import StoryFileReader

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
            description='trains a dialogue model')

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

    parser.add_argument(
            '-o', '--out',
            type=str,
            required=False,
            help="directory to persist the trained model in")
    parser.add_argument(
            '-d', '--domain',
            type=str,
            required=False,
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
            '--interactive',
            default=False,
            action='store_true',
            help="enable interactive training")
    parser.add_argument(
            '--finetune',
            default=False,
            action='store_true',
            help="retrain the model immediately based on feedback.")
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
    parser.add_argument(
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")
    parser.add_argument(
            '--nlu_threshold',
            type=float,
            default=DEFAULT_NLU_FALLBACK_THRESHOLD,
            help="If NLU prediction confidence is below threshold, fallback "
                 "will get triggered.")
    parser.add_argument(
            '--core_threshold',
            type=float,
            default=DEFAULT_CORE_FALLBACK_THRESHOLD,
            help="If Core action prediction confidence is below the threshold "
                 "a fallback action will get triggered")
    parser.add_argument(
            '--fallback_action_name',
            type=str,
            default=DEFAULT_FALLBACK_ACTION,
            help="When a fallback is triggered (e.g. because the ML prediction "
                 "is of low confidence) this is the name of tje action that "
                 "will get triggered instead.")
    parser.add_argument(
            '--mode',
            type=str,
            default="default",
            help="default|compare (train a model, or train multiple models to "
                 "compare policies)")
    # the arguments below are for the compare mode of the script
    parser.add_argument(
            '--epochs_embed',
            type=int,
            default=1000,
            help="number of epochs for the embedding policy")
    parser.add_argument(
            '--epochs_keras',
            type=int,
            default=400,
            help="number of epochs for the keras policy")
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

    utils.add_logging_option_arguments(parser)
    return parser


def train_dialogue_model(domain_file, stories_file, output_path,
                         interpreter=None,
                         endpoints=AvailableEndpoints(),
                         max_history=None,
                         dump_flattened_stories=False,
                         kwargs=None):
    if not kwargs:
        kwargs = {}

    fallback_args, kwargs = utils.extract_args(kwargs,
                                               {"nlu_threshold",
                                                "core_threshold",
                                                "fallback_action_name"})

    policies = [
        FallbackPolicy(
                fallback_args.get("nlu_threshold",
                                  DEFAULT_NLU_FALLBACK_THRESHOLD),
                fallback_args.get("core_threshold",
                                  DEFAULT_CORE_FALLBACK_THRESHOLD),
                fallback_args.get("fallback_action_name",
                                  DEFAULT_FALLBACK_ACTION)),
        MemoizationPolicy(
                max_history=max_history),
        KerasPolicy(
                MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                            max_history=max_history))]

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

    training_data = agent.load_data(stories_file, **data_load_args)
    agent.train(training_data, **kwargs)
    agent.persist(output_path, dump_flattened_stories)

    return agent


def train_comparison_models(story_filename,
                            domain,
                            epochs,
                            output_path=None,
                            exclusion_percentage=None,
                            starspace=True,
                            max_history=None):

    """Trains either a KerasPolicy model or an EmbeddingPolicy, excluding a
    certain percentage of a story file"""

    if starspace:
        featurizer = FullDialogueTrackerFeaturizer(
                        LabelTokenizerSingleStateFeaturizer())
        policies = [EmbeddingPolicy()]
    else:
        featurizer = MaxHistoryTrackerFeaturizer(
                        BinarySingleStateFeaturizer(),
                        max_history=max_history)
        policies = [KerasPolicy(featurizer)]

    agent = Agent(domain,
                  policies=policies)

    data = agent.load_data(story_filename,
                           remove_duplicates=True,
                           augmentation_factor=0,
                           exclusion_percentage=exclusion_percentage)

    agent.train(data,
                rnn_size=64,
                epochs=epochs,
                embed_dim=20,
                attn_shift_range=5)

    agent.persist(model_path=output_path)


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

    utils.configure_colored_logging(cmdline_args.loglevel)

    additional_arguments = {
        "epochs": cmdline_args.epochs,
        "batch_size": cmdline_args.batch_size,
        "validation_split": cmdline_args.validation_split,
        "augmentation_factor": cmdline_args.augmentation,
        "debug_plots": cmdline_args.debug_plots,
        "nlu_threshold": cmdline_args.nlu_threshold,
        "core_threshold": cmdline_args.core_threshold,
        "fallback_action_name": cmdline_args.fallback_action_name
    }

    if cmdline_args.url:
        stories = utils.download_file_from_url(cmdline_args.url)
    else:
        stories = cmdline_args.stories

    _endpoints = AvailableEndpoints.read_endpoints(cmdline_args.endpoints)
    _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                     _endpoints.nlu)

    if cmdline_args.core and cmdline_args.mode == 'default':
        if not cmdline_args.interactive:
            raise ValueError("--core can only be used together with the"
                             "--interactive flag.")
        elif cmdline_args.finetune:
            raise ValueError("--core can only be used together with the"
                             "--interactive flag and without --finetune flag.")
        else:
            logger.info("loading a pre-trained model. ",
                        "all training-related parameters will be ignored")
        _agent = Agent.load(cmdline_args.core,
                            interpreter=_interpreter,
                            generator=_endpoints.nlg,
                            action_endpoint=_endpoints.action)
    elif cmdline_args.mode == 'default':
        if not cmdline_args.out:
            raise ValueError("you must provide a path where the model "
                             "will be saved using -o / --out")
        _agent = train_dialogue_model(cmdline_args.domain,
                                      stories,
                                      cmdline_args.out,
                                      _interpreter,
                                      _endpoints,
                                      cmdline_args.history,
                                      cmdline_args.dump_stories,
                                      additional_arguments)

    elif cmdline_args.mode == 'compare':
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

                train_comparison_models(story_filename=cmdline_args.stories,
                                        domain=cmdline_args.domain,
                                        epochs=cmdline_args.epochs_embed,
                                        output_path=output_path_embed,
                                        exclusion_percentage=i,
                                        starspace=True)

                logger.info("Finished training embed round {}/{}".format(
                                                    current_round,
                                                    len(cmdline_args.percentages)))

                logging.info("Starting to train keras round {}/{}".format(
                                                    current_round,
                                                    len(cmdline_args.percentages)))

                train_comparison_models(story_filename=cmdline_args.stories,
                                        domain=cmdline_args.domain,
                                        epochs=cmdline_args.epochs_keras,
                                        output_path=output_path_keras,
                                        exclusion_percentage=i,
                                        starspace=False)

                logger.info("Finished training keras round {}/{}".format(
                                                    current_round,
                                                    len(cmdline_args.percentages)))

        no_stories = get_no_of_stories(cmdline_args.stories, cmdline_args.domain)

        # store the list of the number of stories present at each exclusion
        # percentage
        story_range = [no_stories - round((x/100.0) * no_stories) for x in
                       cmdline_args.percentages]

        pickle.dump(story_range,
                    io.open(os.path.join(cmdline_args.out, 'num_stories.p'), 'wb'))

    else:
        raise ValueError("--mode can take the values default or compare")

    if cmdline_args.interactive:
        interactive.run_interactive_learning(_agent, finetune=cmdline_args.finetune)
