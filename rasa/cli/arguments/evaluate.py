import argparse
from rasa.cli.arguments.default_arguments import add_endpoint_param, add_domain_param


def set_markers_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa evaluate markers`"""
    parser.add_argument(
        "output_filename",
        type=str,
        help="The filename to write the extracted markers to (CSV format).",
    )

    parser.add_argument(
        "--config",
        default="markers.yml",
        type=str,
        help="The config file(s) containing marker definitions. This can be a single "
        "YAML file, or a directory that contains several files with domain "
        "specifications in it. The content of these files will be read and merged "
        "together.",
    )

    stats = parser.add_mutually_exclusive_group()

    stats.add_argument(
        "--no-stats",
        default=False,
        action="store_true",
        dest="stats",
        help="Do not compute summary statistics.",
    )

    stats.add_argument(
        "--stats-file",
        default="stats.json",
        type=str,
        help="The filename to write out computed summary statistics.",
    )

    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )

    add_domain_param(parser)


def set_markers_first_n_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa evaluate markers by_first_n`"""
    parser.add_argument(
        "count", type=int, help="The number of trackers to extract markers from",
    )


def set_markers_sample_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa evaluate markers by_sample"""
    parser.add_argument(
        "--seed", type=int, help="Seed to use if selecting trackers by 'sample'"
    )
    parser.add_argument(
        "count", type=int, help="The number of trackers to extract markers from",
    )


def set_markers_all_arguments(parser: argparse.ArgumentParser):
    pass


def set_evaluate_arguments(parser: argparse.ArgumentParser):
    # Two subparsers
    pass
