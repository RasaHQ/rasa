import argparse
from typing import List, TextIO

from rasa.core.utils import AvailableEndpoints
from rasa.core.tracker_store import TrackerStore
from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader

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
        help="Extract markers from trackers in a tracker store.",
    )
    markers_arguments.set_markers_extract_arguments(marker_extract_parser)

    marker_extract_parser.set_defaults(func=_run_extract)

    marker_stats_parser = markers_subparser.add_parser(
        "stats",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Compute stats from previously extracted markers.",
    )
    markers_arguments.set_markers_stats_arguments(marker_stats_parser)

    marker_stats_parser.set_defaults(func=_run_stats)


def _run_extract(args: argparse.Namespace):
    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)
    tracker_store = TrackerStore.create(endpoints.tracker_store)
    tracker_loader = MarkerTrackerLoader(
        tracker_store, args.strategy, args.count, args.seed
    )
    trackers = tracker_loader.load()
    # markers = ConversationMarkers.from_trackers(trackers)
    # markers.to_file(args.output_filename)
    # if args.stats:
    #     _stats(None, args.stats)
    pass


def _run_stats(args: argparse.Namespace):
    # markers = ConversationMarkers.from_file(args.input_filename)
    # _stats(markers, args.output_filename)
    pass


def _stats(markers, out_file: TextIO):
    # TODO: Figure out how this is done
    pass
