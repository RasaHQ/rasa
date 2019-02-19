import argparse
from typing import Text

from rasa.model import DEFAULT_MODELS_PATH


def add_model_param(parser: argparse.ArgumentParser, model_name: Text = "Rasa"
                    ) -> None:
    parser.add_argument("-m", "--model",
                        type=str,
                        default=DEFAULT_MODELS_PATH,
                        help="Path to a trained {} model. If a directory "
                             "is specified, it will use the latest model "
                             "in this directory.".format(model_name))


def add_stories_param(parser: argparse.ArgumentParser,
                      stories_name: Text = "training") -> None:
    parser.add_argument(
        "-s", "--stories",
        type=str,
        default="data/core",
        help="File or folder containing {} stories.".format(stories_name))


def add_domain_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-d", "--domain",
                        type=str,
                        default="domain.yml",
                        help="Domain specification (yml file)")


def add_config_param(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yml",
        help="The policy and NLU pipeline configuration of your bot.")
