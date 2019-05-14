import argparse
import asyncio
import os
from typing import List

from rasa import data
from rasa.cli.arguments.default_arguments import (
    add_config_param,
    add_domain_param,
    add_stories_param,
)
from rasa.cli.arguments import show as arguments
from rasa.constants import DEFAULT_DATA_PATH


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    show_parser = subparsers.add_parser(
        "show",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Visualize Rasa Stack data",
    )
    show_parser.set_defaults(func=lambda _: show_parser.print_help(None))

    show_subparsers = show_parser.add_subparsers()
    show_stories_subparser = show_subparsers.add_parser(
        "stories",
        conflict_handler="resolve",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Show Rasa Core stories",
    )
    show_stories_subparser.set_defaults(func=show_stories)

    arguments.set_show_stories_arguments(show_stories_subparser)


def show_stories(args: argparse.Namespace):
    import rasa.core.visualize

    loop = asyncio.get_event_loop()
    args.config = args.config
    args.url = None

    args.stories = data.get_core_directory(args.stories)
    if os.path.exists(DEFAULT_DATA_PATH):
        args.nlu_data = data.get_nlu_directory(DEFAULT_DATA_PATH)

    loop.run_until_complete(
        rasa.core.visualize(
            args.config,
            args.domain,
            args.stories,
            args.nlu_data,
            args.output,
            args.max_history,
        )
    )
