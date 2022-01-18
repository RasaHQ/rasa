import argparse
from pathlib import Path
from rasa.cli.arguments.default_arguments import add_endpoint_param, add_domain_param


def set_markers_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies arguments for `rasa evaluate markers`."""
    parser.add_argument(
        "output_filename",
        type=Path,
        help="The filename to write the extracted markers to (CSV format).",
    )

    parser.add_argument(
        "--config",
        default="markers.yml",
        type=Path,
        help="The marker configuration file(s) containing marker definitions. "
        "This can be a single YAML file, or a directory that contains several "
        "files with marker definitions in it. The content of these files will "
        "be read and merged together.",
    )

    stats = parser.add_mutually_exclusive_group()

    stats.add_argument(
        "--no-stats",
        action="store_false",
        dest="stats",
        help="Do not compute summary statistics.",
    )

    stats.add_argument(
        "--stats-file-prefix",
        default="stats",
        nargs="?",
        type=Path,
        help="The common file prefix of the files where we write out the compute "
        "statistics. More precisely, the file prefix must consist of a common "
        "path plus a common file prefix, to which suffixes `-overall.csv` and "
        "`-per-session.csv` will be added automatically.",
    )

    add_endpoint_param(
        parser, help_text="Configuration file for the tracker store as a yml file."
    )

    add_domain_param(parser)


def set_markers_first_n_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies arguments for `rasa evaluate markers first_n`."""
    parser.add_argument(
        "count", type=int, help="The number of trackers to extract markers from"
    )


def set_markers_sample_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies arguments for `rasa evaluate markers sample_n`."""
    parser.add_argument(
        "--seed", type=int, help="Seed to use if selecting trackers by 'sample_n'"
    )
    parser.add_argument(
        "count", type=int, help="The number of trackers to extract markers from"
    )
