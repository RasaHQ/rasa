import argparse
from typing import List

from rasa.cli import SubParsersAction
import rasa.cli.arguments.markers as markers_arguments


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all markers parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    marker_parser = subparsers.add_parser(
        "marker",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Applies marker conditions to existing trackers.",
    )

    markers_arguments.set_markers_arguments(marker_parser)

    markers_subparser = marker_parser.add_subparsers()
    marker_extract_parser = markers_subparser.add_parser(
        "extract",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Extract markers.",
    )
    markers_arguments.set_markers_extract_arguments(marker_extract_parser)

    # TODO implement run_markers_extract
    #  marker_extract_parser.set_defaults(func=run_markers_extract)

    marker_stats_parser = markers_subparser.add_parser(
        "stats",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Compute stats.",
    )
    markers_arguments.set_markers_stats_arguments(marker_stats_parser)

    # TODO implement run_markers_stats
    #  marker_stats_parser.set_defaults(func=run_markers_stats)
