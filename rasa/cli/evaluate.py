import argparse
from typing import List, Text, Optional

from rasa.core.utils import AvailableEndpoints
from rasa.core.tracker_store import TrackerStore
from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader
from rasa.core.evaluation.marker_base import Marker

from rasa.cli import SubParsersAction
import rasa.cli.arguments.evaluate as arguments
import rasa.shared.utils.cli
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

    markers_subparser.add_parser(
        "all",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select all trackers.",
    )

    marker_parser.set_defaults(func=_run_markers_cli)


def _run_markers_cli(args: argparse.Namespace) -> None:
    """Run markers algorithm using parameters from CLI.

    Args:
        args: The arguments passed in from the CLI.
    """
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
    seed: Optional[int],
    count: Optional[int],
    endpoint_config: Text,
    strategy: Text,
    config: Text,
    output_filename: Text,
    stats_file: Optional[Text] = None,
):
    """Run markers algorithm over specified config and tracker store.

    Args:
        seed: (Optional) The seed to initialise the random number generator for
              use with the 'sample' strategy.
        count: (Optional) Number of trackers to extract from (for any strategy
               except 'all').
        endpoint_config: Path to the endpoint configuration defining the tracker
                         store to use.
        strategy: Strategy to use when selecting trackers to extract from.
        config: Path to the markers definition file to use.
        output_filename: Path to write out the extracted markers.
        stats_file: (Optional) Path to write out statistics about the extracted
                    markers.
    """

    if os.path.exists(output_filename):
        rasa.shared.utils.cli.print_error_and_exit(
            "A file with the output filename already exists"
        )

    if stats_file and os.path.exists(stats_file):
        rasa.shared.utils.cli.print_error_and_exit(
            "A file with the stats filename already exists"
        )

    tracker_loader = _create_tracker_loader(endpoint_config, strategy, count, seed)
    markers = Marker.from_path(config)
    markers.export_markers(tracker_loader, output_filename, stats_file)


def _create_tracker_loader(
    endpoint_config: Text, strategy: Text, count: Optional[int], seed: Optional[int]
) -> MarkerTrackerLoader:
    """Create a tracker loader against the configured tracker store.

    Args:
        endpoint_config: Path to the endpoint configuration defining the tracker
                         store to use.
        strategy: Strategy to use when selecting trackers to extract from.
        count: (Optional) Number of trackers to extract from (for any strategy
               except 'all').
        seed: (Optional) The seed to initialise the random number generator for
              use with the 'sample' strategy.

    Returns:
        A MarkerTrackerLoader object configured with the specified strategy against
        the configured tracker store.
    """
    endpoints = AvailableEndpoints.read_endpoints(endpoint_config)
    tracker_store = TrackerStore.create(endpoints.tracker_store)
    return MarkerTrackerLoader(tracker_store, strategy, count, seed,)
