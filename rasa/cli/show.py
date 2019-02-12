import argparse
import os

import rasa_core.visualize as visualize

from rasa.cli.default_arguments import add_domain_param, add_stories_param, \
    add_config_param
from rasa.model import DEFAULTS_NLU_DATA_PATH


def add_subparser(subparsers, parents):
    show_parser = subparsers.add_parser(
        "show",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Visualize Rasa Stack data.")

    show_subparsers = show_parser.add_subparsers()
    show_stories_subparser = show_subparsers.add_parser(
        "stories",
        conflict_handler='resolve',
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Show Rasa Core stories")

    add_core_visualization_params(show_stories_subparser)
    add_config_param(show_stories_subparser)
    show_stories_subparser.set_defaults(func=show)


def add_core_visualization_params(parser):
    visualize.add_visualization_arguments(parser)
    add_domain_param(parser)
    add_stories_param(parser)


def show(args):
    args.config = [args.config]
    args.url = None

    if os.path.isdir(DEFAULTS_NLU_DATA_PATH):
        args.nlu = DEFAULTS_NLU_DATA_PATH

    visualize.visualize(args)
