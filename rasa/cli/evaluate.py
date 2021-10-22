import argparse
from typing import List, TextIO, Text, Dict, Union

from rasa.core.utils import AvailableEndpoints
from rasa.core.tracker_store import TrackerStore
from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader

from rasa.cli import SubParsersAction
import rasa.cli.arguments.evaluate as arguments
import json
import os.path


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

    marker_parser.set_defaults(func=_run_markers)


def _run_markers_cli(args: argparse.Namespace):
    seed = args.seed if "seed" in args else None
    count = args.count if "count" in args else None

    _run_markers(
        seed, count, args.endpoints, args.strategy, args.config, args.output_filename
    )


def _run_markers(
    seed: int,
    count: int,
    endpoint_config: Text,
    strategy: Text,
    config: Text,
    output_filename: Text,
):
    tracker_loader = _create_tracker_loader(endpoint_config, strategy, count, seed)
    markers = _load_markers(config)

    results = _collect_markers(markers, tracker_loader)
    _save_results(output_filename, results)

    # if args.stats:
    #     _stats(None, args.stats)
    pass


def _load_markers(confpath: Text) -> "Markers":
    return Markers.from_path(confpath)


def _create_tracker_loader(
    endpoint_config: Text, strategy: Text, count: int, seed: int
) -> MarkerTrackerLoader:
    endpoints = AvailableEndpoints.read_endpoints(endpoint_config)
    tracker_store = TrackerStore.create(endpoints.tracker_store)
    return MarkerTrackerLoader(tracker_store, strategy, count, seed,)


def _collect_markers(
    markers: "Markers", tracker_loader: MarkerTrackerLoader
) -> List[Union[Text, Dict[Text, "EvaluationResult"]]]:
    processed_trackers = []

    for tracker in tracker_loader.load():
        tracker_result = markers.evaluate_events(tracker.events)
        tracker_result["sender_id"] = tracker.sender_id
        processed_trackers.append(tracker_result)

    return processed_trackers


def _save_results(
    path: Text, results: List[Union[Text, Dict[Text, "EvaluationResult"]]]
) -> None:
    with open(path, "w") as f:
        f.write(json.dumps(results))


def _compute_stats(
    results: List[Union[Text, Dict[Text, "EvaluationResult"]]], out_file: str
):
    # TODO: Figure out how this is done
    pass
