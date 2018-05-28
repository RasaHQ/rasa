from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import json
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from rasa_core import utils
from rasa_core.evaluate import collect_story_predictions
from rasa_nlu import utils as nlu_utils

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for the evaluate script."""

    parser = argparse.ArgumentParser(
            description='evaluates a dialogue model')
    parser.add_argument(
            '--models',
            type=str,
            required=True,
            help="folder containing trained models"
    )
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing stories to evaluate on")
    parser.add_argument(
            '-o', '--output',
            type=str,
            default="plot.pdf",
            help="output path for the created evaluation plot. If set to None"
                 "or an empty string, no plot will be generated.")

    utils.add_logging_option_arguments(parser)
    return parser


def run_comparison_evaluation(models, stories, output):

    num_correct = defaultdict(list)

    for run in nlu_utils.list_subdirectories(models):
        correct_embed = []
        correct_keras = []
        for model in nlu_utils.list_subdirectories(run):
            actual, preds, failed_stories = collect_story_predictions(stories,
                                                                      model)
            if 'keras' in model:
                correct_keras.append(len(actual) - len(failed_stories))
            elif 'embed' in model:
                correct_embed.append(len(actual) - len(failed_stories))
        num_correct['keras'].append(correct_keras)
        num_correct['embed'].append(correct_embed)

    with open(output + 'results.json', 'w') as f:
        json.dump(num_correct, f)


def plot_curve(filename, no_stories, ax=None, **kwargs):
    ax = ax or plt.gca()
    with open(filename) as f:
        data = json.load(f)
    x = no_stories
    for label in ['keras', 'embed']:
        if len(data[label]) == 0:
            continue
        mean = np.mean(data[label], axis=0)
        std = np.std(data[label], axis=0)
        ax.plot(x, mean, label=label, marker='.')
        ax.fill_between(x,
                        [m-s for m, s in zip(mean, std)],
                        [m+s for m, s in zip(mean, std)],
                        color='#6b2def',
                        alpha=0.2)
    ax.legend(loc=4)
    plt.show()


if __name__ == '__main__':
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    utils.configure_colored_logging(cmdline_args.loglevel)

    run_comparison_evaluation(cmdline_args.models, cmdline_args.stories,
                              cmdline_args.output)

    no_stories = pickle.load(cmdline_args.models + 'num_stories.p', 'rb')

    plot_curve(cmdline_args.output + 'results.json', no_stories)
