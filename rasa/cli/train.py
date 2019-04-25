import argparse
import tempfile
import typing
from typing import List, Optional, Text

from rasa.cli.default_arguments import (
    add_config_param,
    add_domain_param,
    add_nlu_data_param,
    add_stories_param,
)
from rasa.cli.utils import get_validated_path
from rasa.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Interpreter


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    import rasa.core.cli.train as core_cli

    train_parser = subparsers.add_parser("train", help="Train the Rasa bot")

    train_subparsers = train_parser.add_subparsers()
    train_core_parser = train_subparsers.add_parser(
        "core",
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train Rasa Core",
    )
    train_core_parser.set_defaults(func=train_core)

    train_nlu_parser = train_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Train Rasa NLU",
    )
    train_nlu_parser.set_defaults(func=train_nlu)

    for p in [train_parser, train_core_parser, train_nlu_parser]:
        add_general_arguments(p)

    for p in [train_core_parser, train_parser]:
        add_domain_param(p)
        core_cli.add_general_args(p)
    add_stories_param(train_core_parser)
    _add_core_compare_arguments(train_core_parser)

    add_nlu_data_param(train_nlu_parser)

    add_joint_parser_arguments(train_parser)
    train_parser.set_defaults(func=train)


def add_joint_parser_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a model training even if the data has not changed.",
    )
    parser.add_argument(
        "--data",
        default=[DEFAULT_DATA_PATH],
        nargs="+",
        help="Paths to the Core and NLU training files.",
    )


def add_general_arguments(parser: argparse.ArgumentParser):
    add_config_param(parser)
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=DEFAULT_MODELS_PATH,
        help="Directory where your models are stored",
    )


def _add_core_compare_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--percentages",
        nargs="*",
        type=int,
        default=[0, 5, 25, 50, 70, 90, 95],
        help="Range of exclusion percentages",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of runs for experiments"
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        default=[DEFAULT_CONFIG_PATH],
        help="The policy and NLU pipeline configuration of your bot."
        "If multiple configuration files are provided, multiple dialogue "
        "models are trained to compare policies.",
    )


def train(args: argparse.Namespace) -> Optional[Text]:
    import rasa

    domain = get_validated_path(args.domain, "domain", DEFAULT_DOMAIN_PATH)
    config = get_validated_path(args.config, "config", DEFAULT_CONFIG_PATH)

    training_files = [
        get_validated_path(f, "data", DEFAULT_DATA_PATH) for f in args.data
    ]

    return rasa.train(
        domain,
        config,
        training_files,
        args.out,
        args.force,
        extract_additional_arguments(args),
    )


def train_core(
    args: argparse.Namespace, train_path: Optional[Text] = None
) -> Optional[Text]:
    from rasa.train import train_core
    import asyncio

    loop = asyncio.get_event_loop()
    output = train_path or args.out

    args.domain = get_validated_path(args.domain, "domain", DEFAULT_DOMAIN_PATH)
    stories = get_validated_path(args.stories, "stories", DEFAULT_DATA_PATH)

    _train_path = train_path or tempfile.mkdtemp()

    # Policies might be a list for the compare training. Do normal training
    # if only list item was passed.
    if not isinstance(args.config, list) or len(args.config) == 1:
        if isinstance(args.config, list):
            args.config = args.config[0]

        config = get_validated_path(args.config, "config", DEFAULT_CONFIG_PATH)

        return train_core(
            args.domain,
            config,
            stories,
            output,
            train_path,
            extract_additional_arguments(args),
        )
    else:
        from rasa.core.train import do_compare_training

        loop.run_until_complete(do_compare_training(args, stories, None))
        return None


def train_nlu(
    args: argparse.Namespace, train_path: Optional[Text] = None
) -> Optional["Interpreter"]:
    from rasa.train import train_nlu

    output = train_path or args.out

    config = get_validated_path(args.config, "config", DEFAULT_CONFIG_PATH)
    nlu_data = get_validated_path(args.nlu, "nlu", DEFAULT_DATA_PATH)

    return train_nlu(config, nlu_data, output, train_path)


def extract_additional_arguments(args: argparse.Namespace) -> typing.Dict:
    return {
        "augmentation_factor": args.augmentation,
        "dump_stories": args.dump_stories,
        "debug_plots": args.debug_plots,
        "percentages": args.percentages,
        "runs": args.runs,
    }
