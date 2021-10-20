import argparse
from rasa.cli.arguments.default_arguments import add_endpoint_param


def set_markers_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa markers extract`"""
    parser.add_argument(
        "--by",
        choices=["first-n", "sample", "all"],
        required=True,
        dest="strategy",
        help="The strategy used to select trackers for marker extraction",
    )

    parser.add_subparsers()

    parser.add_argument(
        "output_filename",
        type=argparse.FileType("w"),
        help="The filename to write the extracted markers to",
    )

    stats = parser.add_mutually_exclusive_group()

    stats.add_argument(
        "--no-stats",
        default=True,
        action="store_false",
        dest="stats",
        help="Do not compute summary statistics",
    )

    stats.add_argument(
        "--stats-file",
        type=argparse.FileType("w"),
        help="Che filename to write out computed summary statistics",
    )

    first_n_arguments = parser.add_argument_group("Arguments for strategy 'by_first_n'")
    set_markers_first_n_arguments(first_n_arguments)

    sample_arguments = parser.add_argument_group("Arguments for strategy 'by_sample'")
    set_markers_sample_arguments(sample_arguments)

    add_endpoint_param(parser)


def set_markers_first_n_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa markers by_first_n`"""
    parser.add_argument(
        "--num-trackers",
        type=int,
        dest="count",
        help="The number of trackers to extract markers from",
    )


def set_markers_sample_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa markers by_sample"""
    parser.add_argument(
        "--seed", type=int, help="Seed to use if selecting trackers by 'sample'"
    )
    parser.add_argument(
        "--num-trackers",
        type=int,
        dest="count",
        help="The number of trackers to extract markers from",
    )


def set_markers_all_arguments(parser: argparse.ArgumentParser):
    pass


def set_evaluate_arguments(parser: argparse.ArgumentParser):
    # Two subparsers
    pass
