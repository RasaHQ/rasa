import argparse
from typing import List, Text, Optional
from pathlib import Path

from rasa import telemetry
from rasa.core.utils import AvailableEndpoints
from rasa.core.tracker_store import TrackerStore
from rasa.core.evaluation.marker_tracker_loader import MarkerTrackerLoader
from rasa.core.evaluation.marker_base import Marker, OperatorMarker
from rasa.shared.core.domain import Domain
from rasa.cli import SubParsersAction
import rasa.cli.arguments.evaluate as arguments
import rasa.shared.utils.cli

STATS_OVERALL_SUFFIX = "-overall.csv"
STATS_SESSION_SUFFIX = "-per-session.csv"


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

    markers_subparser = marker_parser.add_subparsers(dest="strategy")

    markers_first_n_subparser = markers_subparser.add_parser(
        "first_n",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select trackers sequentially until N are taken.",
    )
    arguments.set_markers_first_n_arguments(markers_first_n_subparser)

    arguments.set_markers_arguments(markers_first_n_subparser)

    markers_sample_subparser = markers_subparser.add_parser(
        "sample_n",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select trackers by sampling N.",
    )
    arguments.set_markers_sample_arguments(markers_sample_subparser)

    arguments.set_markers_arguments(markers_sample_subparser)

    markers_all_subparser = markers_subparser.add_parser(
        "all",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Select all trackers.",
    )

    arguments.set_markers_arguments(markers_all_subparser)

    marker_parser.set_defaults(func=_run_markers_cli)


def _run_markers_cli(args: argparse.Namespace) -> None:
    """Run markers algorithm using parameters from CLI.

    Args:
        args: The arguments passed in from the CLI.
    """
    seed = args.seed if "seed" in args else None
    count = args.count if "count" in args else None

    stats_file_prefix = args.stats_file_prefix if args.stats else None

    _run_markers(
        seed,
        count,
        args.endpoints,
        args.domain,
        args.strategy,
        args.config,
        args.output_filename,
        stats_file_prefix,
    )


def _run_markers(
    seed: Optional[int],
    count: Optional[int],
    endpoint_config: Path,
    domain_path: Optional[Text],
    strategy: Text,
    config: Path,
    output_filename: Path,
    stats_file_prefix: Optional[Path] = None,
) -> None:
    """Run markers algorithm over specified config and tracker store.

    Args:
        seed: (Optional) The seed to initialise the random number generator for
              use with the 'sample' strategy.
        count: (Optional) Number of trackers to extract from (for any strategy
               except 'all').
        endpoint_config: Path to the endpoint configuration defining the tracker
                         store to use.
        domain_path: Path to the domain specification to use when validating the
                     marker definitions.
        strategy: Strategy to use when selecting trackers to extract from.
        config: Path to the markers definition file to use.
        output_filename: Path to write out the extracted markers.
        stats_file_prefix: (Optional) A prefix used to create paths where files with
            statistics on the marker extraction results will be written.
            It must consists of the path to the where those files should be stored
            and the common file prefix, e.g. '<path-to-stats-folder>/statistics'.
            Statistics derived from all marker extractions will be stored in
            '<path-to-stats-folder>/statistics-overall.csv', while the statistics
            computed per session will be stored in
            '<path-to-stats-folder>/statistics-per-session.csv'.
    """
    telemetry.track_markers_extraction_initiated(
        strategy=strategy,
        only_extract=stats_file_prefix is not None,
        seed=seed is not None,
        count=count,
    )

    domain = Domain.load(domain_path) if domain_path else None
    markers = Marker.from_path(config)
    if domain and not markers.validate_against_domain(domain):
        rasa.shared.utils.cli.print_error_and_exit(
            "Validation errors were found in the markers definition. "
            "Please see errors listed above and fix before running again."
        )

    # Calculate telemetry
    # All loaded markers are combined with one virtual OR over all markers
    num_markers = len(markers.sub_markers)
    max_depth = markers.max_depth() - 1
    # Find maximum branching of marker
    branching_factor = max(
        (
            len(sub_marker.sub_markers)
            for marker in markers.sub_markers
            for sub_marker in marker.flatten()
            if isinstance(sub_marker, OperatorMarker)
        ),
        default=0,
    )

    telemetry.track_markers_parsed_count(num_markers, max_depth, branching_factor)

    tracker_loader = _create_tracker_loader(
        endpoint_config, strategy, domain, count, seed
    )

    def _append_suffix(path: Optional[Path], suffix: Text) -> Optional[Path]:
        return path.parent / (path.name + suffix) if path else None

    try:
        import asyncio

        asyncio.run(
            markers.evaluate_trackers(
                trackers=tracker_loader.load(),
                output_file=output_filename,
                session_stats_file=_append_suffix(
                    stats_file_prefix, STATS_SESSION_SUFFIX
                ),
                overall_stats_file=_append_suffix(
                    stats_file_prefix, STATS_OVERALL_SUFFIX
                ),
            )
        )
    except (FileExistsError, NotADirectoryError) as e:
        rasa.shared.utils.cli.print_error_and_exit(message=str(e))


def _create_tracker_loader(
    endpoint_config: Text,
    strategy: Text,
    domain: Domain,
    count: Optional[int],
    seed: Optional[int],
) -> MarkerTrackerLoader:
    """Create a tracker loader against the configured tracker store.

    Args:
        endpoint_config: Path to the endpoint configuration defining the tracker
                         store to use.
        strategy: Strategy to use when selecting trackers to extract from.
        domain: The domain to use when connecting to the tracker store.
        count: (Optional) Number of trackers to extract from (for any strategy
               except 'all').
        seed: (Optional) The seed to initialise the random number generator for
              use with the 'sample_n' strategy.

    Returns:
        A MarkerTrackerLoader object configured with the specified strategy against
        the configured tracker store.
    """
    endpoints = AvailableEndpoints.read_endpoints(endpoint_config)
    tracker_store = TrackerStore.create(endpoints.tracker_store, domain=domain)
    return MarkerTrackerLoader(tracker_store, strategy, count, seed)
