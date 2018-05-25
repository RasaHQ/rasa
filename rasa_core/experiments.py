from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io
import logging
import uuid
from difflib import SequenceMatcher

from builtins import str
from tqdm import tqdm
from typing import Text, List, Tuple

import rasa_core
from rasa_core import training
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.events import ActionExecuted, UserUttered
from rasa_core.interpreter import RegexInterpreter, RasaNLUInterpreter
from rasa_core.training.generator import TrainingDataGenerator
from rasa_nlu.evaluate import plot_confusion_matrix, log_evaluation_table

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for the evaluate script."""

    parser = argparse.ArgumentParser(
            description='evaluates a dialogue model')
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing stories to evaluate on")
    parser.add_argument(
            '-o', '--output',
            type=str,
            default="story_confmat.pdf",
            help="output path for the created evaluation plot. If set to None"
                 "or an empty string, no plot will be generated.")

    utils.add_logging_option_arguments(parser)
    return parser


def run_comparison_evaluation(models, stories, runs, percentages):
    count = 0
    for model in models:
        while count < runs:
            for p in percentages:
                actual, preds, failed_stories = run_story_evaluation(stories, )
