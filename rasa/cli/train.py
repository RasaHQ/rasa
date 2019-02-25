import argparse
import tempfile
from typing import List, Text, Optional

from rasa.cli.default_arguments import (add_config_param, add_domain_param,
                                        add_stories_param, add_nlu_data_param)
from rasa.cli.utils import validate
from rasa.constants import (DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH,
                            DEFAULT_STORIES_PATH, DEFAULT_NLU_DATA_PATH,
                            DEFAULT_MODELS_PATH)


def add_subparser(subparsers: argparse._SubParsersAction,
                  parents: List[argparse.ArgumentParser]):
    from rasa_core.cli.train import add_general_args

    train_parser = subparsers.add_parser(
        "train",
        help="Train the Rasa bot")

    train_subparsers = train_parser.add_subparsers()

    train_core_parser = train_subparsers.add_parser(
        "core",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train Rasa Core")
    train_core_parser.set_defaults(func=train_core)

    train_nlu_parser = train_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train Rasa NLU")
    train_nlu_parser.set_defaults(func=train_nlu)

    for p in [train_parser, train_core_parser, train_nlu_parser]:
        add_general_arguments(p)

    for p in [train_core_parser, train_parser]:
        add_core_arguments(p)
        add_general_args(p)
    _add_core_compare_arguments(train_core_parser)

    for p in [train_nlu_parser, train_parser]:
        add_nlu_data_param(p)

    train_parser.set_defaults(func=train)


def add_general_arguments(parser: argparse.ArgumentParser):
    add_config_param(parser)
    parser.add_argument(
        "-o", "--out",
        type=str,
        default=DEFAULT_MODELS_PATH,
        help="Directory where your models are stored")


def add_core_arguments(parser: argparse.ArgumentParser):
    add_domain_param(parser)
    add_stories_param(parser)


def _add_core_compare_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--percentages",
        nargs="*",
        type=int,
        default=[0, 5, 25, 50, 70, 90, 95],
        help="Range of exclusion percentages")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs for experiments")
    parser.add_argument(
        "-c", "--config",
        nargs='*',
        default=["config.yml"],
        help="The policy and NLU pipeline configuration of your bot."
             "If multiple configuration files are provided, multiple dialogue "
             "models are trained to compare policies.")


def train(args: argparse.Namespace) -> Optional[Text]:
    import rasa
    validate(args, [("domain", DEFAULT_DOMAIN_PATH),
                    ("config", DEFAULT_CONFIG_PATH),
                    ("nlu", DEFAULT_NLU_DATA_PATH),
                    ("stories", DEFAULT_STORIES_PATH)])

    return rasa.train(args.domain, args.config, args.stories, args.nlu,
                      args.out)


def train_core(args: argparse.Namespace, train_path: Optional[Text] = None
               ) -> Optional[Text]:
    from rasa.train import train_core

    output = train_path or args.out

    validate(args, [("domain", DEFAULT_DOMAIN_PATH),
                    ("stories", DEFAULT_STORIES_PATH)])

    _train_path = train_path or tempfile.mkdtemp()

    if not isinstance(args.config, list) or len(args.config) == 1:
        if isinstance(args.config, list):
            args.config = args.config[0]

        validate(args, [("config", DEFAULT_CONFIG_PATH)])

        return train_core(args.domain, args.config, args.stories, output,
                          train_path)
    else:
        from rasa_core.train import do_compare_training
        do_compare_training(args, args.stories, None)
        return None


def train_nlu(args: argparse.Namespace, train_path: Optional[Text] = None
              ) -> Optional["Interpreter"]:
    from rasa.train import train_nlu

    output = train_path or args.out

    validate(args, [("config", DEFAULT_CONFIG_PATH),
                    ("nlu", DEFAULT_NLU_DATA_PATH)])

    return train_nlu(args.config, args.nlu_data, output, train_path)
