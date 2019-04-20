import argparse
from typing import Text

from rasa.constants import DEFAULT_DATA_PATH, DEFAULT_MODELS_PATH


def add_model_param(parser: argparse.ArgumentParser, model_name: Text = "Rasa") -> None:
    defaults = {
        "type": str,
        "help": "Path to a trained {} model. If a directory "
        "is specified, it will use the latest model "
        "in this directory.".format(model_name),
    }
    parser.add_argument("-m", "--model", default=DEFAULT_MODELS_PATH, **defaults)
    parser.add_argument("model-as-positional-argument", nargs="?", **defaults)


def add_stories_param(
    parser: argparse.ArgumentParser, stories_name: Text = "training"
) -> None:
    parser.add_argument(
        "-s",
        "--stories",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="File or folder containing {} stories.".format(stories_name),
    )


def add_nlu_data_param(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-u",
        "--nlu",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="File or folder containing your NLU training data.",
    )


def add_domain_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d",
        "--domain",
        type=str,
        default="domain.yml",
        help="Domain specification (yml file)",
    )


def add_config_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yml",
        help="The policy and NLU pipeline configuration of your bot.",
    )
