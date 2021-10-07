import argparse
from rasa.cli.arguments.default_arguments import add_endpoint_param


def set_markers_extract_arguments(parser: argparse.ArgumentParser):
    """Specifies arguments for `rasa markers extract`"""
    parser.add_argument(
        "--stats",
        type=argparse.FileType("w"),
        help="Compute summary statistics over the extracted markers and write them to a specified file",
    )
    parser.add_argument(
        "--num-trackers",
        type=int,
        dest="count",
        help="The number of trackers to extract markers from",
    )
    parser.add_argument(
        "--by",
        choices=["first-n", "sample", "all"],
        required=True,
        dest="strategy",
        help="The strategy used to select trackers for marker extraction",
    )
    parser.add_argument(
        "--seed", type=int, help="Seed to use if selecting trackers by 'sample'"
    )
    parser.add_argument(
        "output_filename",
        type=argparse.FileType("w"),
        help="The filename to write the extracted markers to",
    )

    add_endpoint_param(parser)


def set_markers_stats_arguments(parser: argparse.ArgumentParser):
    """"Specifies arguments for `rasa markers stats`"""
    parser.add_argument(
        "input_filename",
        type=argparse.FileType("r"),
        help="The filename to read extracted markers from",
    )
    parser.add_argument(
        "output_filename",
        type=argparse.FileType("w"),
        help="The filename to write out computed summary statistics",
    )


def set_markers_arguments(parser: argparse.ArgumentParser):
    # Two subparsers
    pass
