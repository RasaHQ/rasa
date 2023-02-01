import argparse
import os
from typing import List

from rasa.cli import SubParsersAction
from rasa.cli.arguments import visualize as arguments
from rasa.shared.constants import DEFAULT_DATA_PATH


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

    dump_graph_parser = subparsers.add_parser(
        "dump_graph",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Dump data as graph.",
    )
    dump_graph_parser.set_defaults(func=dump_graph)
    arguments.add_domain_param(dump_graph_parser)
    arguments.add_stories_param(dump_graph_parser)
    arguments.add_nlu_data_param(dump_graph_parser, default=None, help_text="File or folder with NLU data")
    arguments.add_out_param(
        dump_graph_parser,
        default="graph.json",
        help_text="Filename of the output path, e.g. 'graph.json'.",
    )

def visualize_stories(args: argparse.Namespace) -> None:
    import rasa.core.visualize

    args.stories = rasa.shared.data.get_core_directory(args.stories)
    if args.nlu is None and os.path.exists(DEFAULT_DATA_PATH):
        args.nlu = rasa.shared.data.get_nlu_directory(DEFAULT_DATA_PATH)

    rasa.core.visualize.visualize(
        args.domain, args.stories, args.nlu, args.out, args.max_history
    )


def dump_graph(args: argparse.Namespace):
    import rasa.core.visualize

    args.stories = rasa.shared.data.get_core_directory(args.stories)
    if args.nlu is None and os.path.exists(DEFAULT_DATA_PATH):
        args.nlu = rasa.shared.data.get_nlu_directory(DEFAULT_DATA_PATH)

    rasa.core.visualize.dump_graph(
        args.domain, args.stories, args.nlu, args.out
    )
