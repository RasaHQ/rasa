import argparse
import os
from typing import List

from rasa.cli.default_arguments import (
    add_config_param, add_domain_param,
    add_stories_param)
from rasa.constants import DEFAULT_NLU_DATA_PATH


def add_subparser(subparsers: argparse._SubParsersAction,
                  parents: List[argparse.ArgumentParser]):
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
    show_stories_subparser.set_defaults(func=show_stories)

    show_parser.set_defaults(func=lambda _: show_parser.print_help(None))


def add_core_visualization_params(parser: argparse.ArgumentParser):
    from rasa_core.cli.visualization import add_visualization_arguments

    add_visualization_arguments(parser)
    add_domain_param(parser)
    add_stories_param(parser)


def show_stories(args: argparse.Namespace):
    import rasa_core.visualize

    args.config = args.config
    args.url = None

    if os.path.isdir(DEFAULT_NLU_DATA_PATH):
        args.nlu = DEFAULT_NLU_DATA_PATH

    rasa_core.visualize(args.config, args.domain, args.stories, args.nlu_data,
                        args.output, args.max_history)
