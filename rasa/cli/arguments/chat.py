import argparse

from rasa.cli.arguments.default_arguments import add_model_param
from rasa.cli.arguments.run import add_server_arguments


def add_share_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share your local assistant with a static URL.",
    )
