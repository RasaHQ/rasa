import argparse
import os
from typing import List

from rasa.cli import SubParsersAction
from rasa.cli.arguments import visualize as arguments
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.data
import rasa.utils.common  # noqa: F401


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all visualization parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    visualize_parser = subparsers.add_parser(
        "visualize",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Visualize stories.",
    )
    visualize_parser.set_defaults(func=visualize_stories)

    arguments.set_visualize_stories_arguments(visualize_parser)


def visualize_stories(args: argparse.Namespace) -> None:
    import rasa.core.visualize

    args.stories = rasa.shared.data.get_core_directory(args.stories)
    if args.nlu is None and os.path.exists(DEFAULT_DATA_PATH):
        args.nlu = rasa.shared.data.get_nlu_directory(DEFAULT_DATA_PATH)

    rasa.utils.common.run_in_loop(
        rasa.core.visualize.visualize(
            args.config, args.domain, args.stories, args.nlu, args.out, args.max_history
        )
    )
