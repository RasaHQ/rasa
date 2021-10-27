import argparse
from typing import List, TextIO, Text, Dict, Union

from rasa.core.utils import AvailableEndpoints
from rasa.core.tracker_store import TrackerStore
from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader
import rasa.core.evaluation.marker
from rasa.core.evaluation.marker_base import Marker, DialogueMetaData

from rasa.cli import SubParsersAction
import rasa.cli.arguments.evaluate as arguments
import csv


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all evaluate parsers.

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
        "markers",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Applies marker conditions to existing trackers.",
    )

    arguments.set_markers_arguments(marker_parser)

    markers_subparser = marker_parser.add_subparsers(dest="strategy")

    markers_first_n_subparser = markers_subparser.add_parser(
        "first_n",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select trackers sequentially until N are taken.",
    )
    arguments.set_markers_first_n_arguments(markers_first_n_subparser)

    markers_sample_subparser = markers_subparser.add_parser(
        "sample",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select trackers by sampling N.",
    )
    arguments.set_markers_sample_arguments(markers_sample_subparser)

    markers_all_subparser = markers_subparser.add_parser(
        "all",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select all trackers.",
    )
    arguments.set_markers_all_arguments(markers_all_subparser)

    marker_parser.set_defaults(func=_run_markers_cli)


def _run_markers_cli(args: argparse.Namespace):
    """Run markers algorithm using parameters from CLI."""
    seed = args.seed if "seed" in args else None
    count = args.count if "count" in args else None

    stats_file = args.stats_file if "stats_file" in args and args.stats else None

    _run_markers(
        seed,
        count,
        args.endpoints,
        args.strategy,
        args.config,
        args.output_filename,
        stats_file,
    )


def _run_markers(
    seed: int,
    count: int,
    endpoint_config: Text,
    strategy: Text,
    config: Text,
    output_filename: Text,
    stats_file: Text = None,
):
    """Run markers algorithm over specified config and tracker store."""
    tracker_loader = _create_tracker_loader(endpoint_config, strategy, count, seed)
    markers = Marker.from_path(config)

    results = _collect_markers(markers, tracker_loader)
    _save_results(output_filename, results)

    if stats_file:
        _compute_stats(results, stats_file)


def _create_tracker_loader(
    endpoint_config: Text, strategy: Text, count: int, seed: int
) -> MarkerTrackerLoader:
    """Create a tracker loader against the configured tracker store."""
    endpoints = AvailableEndpoints.read_endpoints(endpoint_config)
    tracker_store = TrackerStore.create(endpoints.tracker_store)
    return MarkerTrackerLoader(tracker_store, strategy, count, seed,)


def _collect_markers(
    markers: Marker, tracker_loader: MarkerTrackerLoader
) -> Dict[Text, List[Dict[Text, DialogueMetaData]]]:
    """Collect markers for each dialogue in each tracker loaded."""
    processed_trackers = {}

    for tracker in tracker_loader.load():
        tracker_result = markers.evaluate_events(tracker.events)
        processed_trackers[tracker.sender_id] = tracker_result

    return processed_trackers


def _save_results(
    path: Text, results: Dict[Text, List[Dict[Text, DialogueMetaData]]]
) -> None:
    """Save extracted marker results as CSV to specified path."""
    with open(path, "w") as f:
        table_writer = csv.writer(f)
        table_writer.writerow(
            [
                "sender_id",
                "dialogue_id",
                "marker_name",
                "event_id",
                "num_preceding_user_turns",
            ]
        )
        for sender_id, dialogues in results.items():
            for dialogue_id, dialogue in enumerate(dialogues):
                for marker_name, marker_metadata in dialogue.items():
                    # TODO: make sure this is updated when timestamp is actually event id
                    for event_id, preceding_user_turns in zip(
                        marker_metadata.event_ids, marker_metadata.preceding_user_turns
                    ):
                        table_writer.writerow(
                            [
                                sender_id,
                                dialogue_id,
                                marker_name,
                                event_id,
                                preceding_user_turns,
                            ]
                        )


def _compute_stats(
    results: List[Union[Text, Dict[Text, DialogueMetaData]]], out_file: str
):
    """Compute stats over extracted marker data."""
    # TODO: Figure out how this is done
    pass
