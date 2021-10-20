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
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tools for evaluating models.",
    )

    evaluate_subparsers = evaluate_parser.add_subparsers()

    marker_parser = evaluate_subparsers.add_parser(
        "marker",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Applies marker conditions to existing trackers.",
    )

    markers_arguments.set_markers_arguments(marker_parser)

    markers_subparser = marker_parser.add_subparsers()

    markers_first_n_subparser = markers_subparser.add_parser(
        "by_first_n",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa models using your test NLU data and stories.",
    )
    markers_arguments.set_markers_first_n_arguments(markers_first_n_subparser)

    markers_sample_subparser = markers_subparser.add_parser(
        "by_sample",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa models using your test NLU data and stories.",
    )
    markers_arguments.set_markers_sample_arguments(markers_sample_subparser)

    markers_all_subparser = markers_subparser.add_parser(
        "by_all",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Tests Rasa models using your test NLU data and stories.",
    )
    markers_arguments.set_markers_all_arguments(markers_all_subparser)

    marker_parser.set_defaults(func=_run_extract)


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


def _stats(markers, out_file: TextIO):
    # TODO: Figure out how this is done
    pass
