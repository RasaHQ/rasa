import argparse
import os
import tempfile
from typing import List, Optional, Text, Dict
import rasa.cli.arguments as arguments

from rasa.cli.utils import (
    get_validated_path,
    missing_config_keys,
    print_error,
    print_warning,
)
from rasa.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
    FALLBACK_CONFIG_PATH,
    CONFIG_MANDATORY_KEYS_NLU,
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS,
)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    import rasa.cli.arguments.train as core_cli

    train_parser = subparsers.add_parser(
        "train",
        help="Trains a Rasa model using your NLU data and stories.",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    arguments.train.set_train_arguments(train_parser)

    train_subparsers = train_parser.add_subparsers()
    train_core_parser = train_subparsers.add_parser(
        "core",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa Core model using your stories.",
    )
    train_core_parser.set_defaults(func=train_core)

    train_nlu_parser = train_subparsers.add_parser(
        "nlu",
        parents=parents,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Trains a Rasa NLU model using your NLU data.",
    )
    train_nlu_parser.set_defaults(func=train_nlu)

    train_parser.set_defaults(func=train)

    arguments.train.set_train_core_arguments(train_core_parser)
    arguments.train.set_train_nlu_arguments(train_nlu_parser)


def train(args: argparse.Namespace) -> Optional[Text]:
    import rasa

    domain = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )

    config = _get_valid_config(args.config, CONFIG_MANDATORY_KEYS)

    training_files = [
        get_validated_path(f, "data", DEFAULT_DATA_PATH, none_is_valid=True)
        for f in args.data
    ]

    return rasa.train(
        domain=domain,
        config=config,
        training_files=training_files,
        output=args.out,
        force_training=args.force,
        fixed_model_name=args.fixed_model_name,
        kwargs=extract_additional_arguments(args),
    )


def train_core(
    args: argparse.Namespace, train_path: Optional[Text] = None
) -> Optional[Text]:
    from rasa.train import train_core
    import asyncio

    loop = asyncio.get_event_loop()
    output = train_path or args.out

    args.domain = get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH, none_is_valid=True
    )
    stories = get_validated_path(
        args.stories, "stories", DEFAULT_DATA_PATH, none_is_valid=True
    )

    _train_path = train_path or tempfile.mkdtemp()

    # Policies might be a list for the compare training. Do normal training
    # if only list item was passed.
    if not isinstance(args.config, list) or len(args.config) == 1:
        if isinstance(args.config, list):
            args.config = args.config[0]

        config = _get_valid_config(args.config, CONFIG_MANDATORY_KEYS_CORE)

        return train_core(
            domain=args.domain,
            config=config,
            stories=stories,
            output=output,
            train_path=train_path,
            fixed_model_name=args.fixed_model_name,
            kwargs=extract_additional_arguments(args),
        )
    else:
        from rasa.core.train import do_compare_training

        loop.run_until_complete(do_compare_training(args, stories, None))
        return None


def train_nlu(
    args: argparse.Namespace, train_path: Optional[Text] = None
) -> Optional[Text]:
    from rasa.train import train_nlu

    output = train_path or args.out

    config = _get_valid_config(args.config, CONFIG_MANDATORY_KEYS_NLU)
    nlu_data = get_validated_path(
        args.nlu, "nlu", DEFAULT_DATA_PATH, none_is_valid=True
    )

    return train_nlu(
        config=config,
        nlu_data=nlu_data,
        output=output,
        train_path=train_path,
        fixed_model_name=args.fixed_model_name,
    )


def extract_additional_arguments(args: argparse.Namespace) -> Dict:
    arguments = {}

    if "augmentation" in args:
        arguments["augmentation_factor"] = args.augmentation
    if "dump_stories" in args:
        arguments["dump_stories"] = args.dump_stories
    if "debug_plots" in args:
        arguments["debug_plots"] = args.debug_plots

    return arguments


def _get_valid_config(
    config: Optional[Text],
    mandatory_keys: List[Text],
    default_config: Text = DEFAULT_CONFIG_PATH,
) -> Text:
    config = get_validated_path(config, "config", default_config)

    if not os.path.exists(config):
        print_error(
            "The config file '{}' does not exist. Use '--config' to specify a "
            "valid config file."
            "".format(config)
        )
        exit(1)

    missing_keys = missing_config_keys(config, mandatory_keys)
    if missing_keys:
        print_error(
            "The config file '{}' is missing mandatory parameters: "
            "'{}'. Add missing parameters to config file and try again."
            "".format(config, "', '".join(missing_keys))
        )
        exit(1)

    return config  # pytype: disable=bad-return-type
