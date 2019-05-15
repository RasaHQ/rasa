import argparse
import asyncio
import os
from typing import List

from rasa import data
from rasa.cli.arguments import visualize as arguments
from rasa.constants import DEFAULT_DATA_PATH


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    visualize_parser = subparsers.add_parser(
        "visualize",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Visualize Rasa Core stories",
    )
    visualize_parser.set_defaults(func=visualize_stories)

    arguments.set_visualize_stories_arguments(visualize_parser)


def visualize_stories(args: argparse.Namespace):
    import rasa.core.visualize

    loop = asyncio.get_event_loop()

    args.stories = data.get_core_directory(args.stories)
    if args.nlu_data is None and os.path.exists(DEFAULT_DATA_PATH):
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
