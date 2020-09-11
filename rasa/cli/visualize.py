import argparse
import os
from typing import List

import rasa.shared.data
from rasa.cli.arguments import visualize as arguments
import rasa.cli.utils
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
        help="Visualize stories.",
    )
    visualize_parser.set_defaults(func=visualize_stories)

    arguments.set_visualize_stories_arguments(visualize_parser)


def visualize_stories(args: argparse.Namespace):
    import rasa.core.visualize

    args.stories = rasa.shared.data.get_core_directory(args.stories)
    if args.nlu is None and os.path.exists(DEFAULT_DATA_PATH):
        args.nlu = rasa.shared.data.get_nlu_directory(DEFAULT_DATA_PATH)

    rasa.cli.utils.run_in_loop(
        rasa.core.visualize(
            args.config, args.domain, args.stories, args.nlu, args.out, args.max_history
        )
    )
