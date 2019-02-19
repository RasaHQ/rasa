import argparse
import os
import tempfile
from typing import List, Text, Optional

import rasa.model as model
from rasa.cli.default_arguments import (add_config_param, add_domain_param,
                                        add_stories_param, add_nlu_data_param)
from rasa.cli.utils import validate, create_default_output_path
from rasa.cli.constants import (DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH,
                                DEFAULT_STORIES_PATH, DEFAULT_NLU_DATA_PATH)
from rasa.model import (
    DEFAULT_MODELS_PATH, core_fingerprint_changed,
    fingerprint_from_path, get_latest_model, merge_model, model_fingerprint,
    nlu_fingerprint_changed, unpack_model)


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
        default=None,
        help="Directory where your models are stored.")


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


def train(args: argparse.Namespace) -> Text:
    from rasa_core.utils import print_success

    validate(args, [("out", DEFAULT_MODELS_PATH, True),
                    ("domain", DEFAULT_DOMAIN_PATH),
                    ("config", DEFAULT_CONFIG_PATH),
                    ("nlu", DEFAULT_NLU_DATA_PATH),
                    ("stories", DEFAULT_STORIES_PATH)])

    output = args.out or create_default_output_path()
    train_path = tempfile.mkdtemp()
    old_model = get_latest_model(output)
    retrain_core = True
    retrain_nlu = True

    new_fingerprint = model_fingerprint(args.config, args.domain, args.nlu,
                                        args.stories)
    if old_model:
        unpacked, old_core, old_nlu = unpack_model(old_model,
                                                   subdirectories=True)
        last_fingerprint = fingerprint_from_path(unpacked)

        if not core_fingerprint_changed(last_fingerprint, new_fingerprint):
            target_path = os.path.join(train_path, "rasa_model", "core")
            retrain_core = merge_model(old_core, target_path)

        if not nlu_fingerprint_changed(last_fingerprint, new_fingerprint):
            target_path = os.path.join(train_path, "rasa_model", "nlu")
            retrain_nlu = merge_model(old_nlu, target_path)

    if retrain_core:
        train_core(args, train_path)
    else:
        print("Core configuration did not change. No need to retrain "
              "Core model.")

    if retrain_nlu:
        train_nlu(args, train_path)
    else:
        print("NLU configuration did not change. No need to retrain NLU model.")

    if retrain_core or retrain_nlu:
        model.create_package_rasa(train_path, "rasa_model", output,
                                  new_fingerprint)

        print("Train path: '{}'.".format(train_path))

        print_success("Your bot is trained and ready to take for a spin!")

        return output
    else:
        print("Nothing changed. You can use the old model: '{}'."
              "".format(old_model))

        return old_model


def train_core(args: argparse.Namespace, train_path: Optional[Text] = None
               ) -> Optional[Text]:
    import rasa_core.train
    from rasa_core.utils import print_success

    args.out = train_path or args.out

    validate(args, [("domain", DEFAULT_DOMAIN_PATH),
                    ("stories", DEFAULT_STORIES_PATH),
                    ("out", DEFAULT_MODELS_PATH, True)])

    _train_path = train_path or tempfile.mkdtemp()

    if not isinstance(args.config, list) or len(args.config) == 1:
        if isinstance(args.config, list):
            args.config = args.config[0]

        validate(args, [("config", DEFAULT_CONFIG_PATH)])

        # normal (not compare) training
        core_model = rasa_core.train.train_dialogue_model(
            domain_file=args.domain,
            stories_file=args.stories,
            output_path=os.path.join(_train_path, "rasa_model", "core"),
            policy_config=args.config)

        if not train_path:
            # Only Core was trained.
            output_path = args.out or create_default_output_path(prefix="core-")
            new_fingerprint = model_fingerprint(args.config, args.domain,
                                                stories=args.stories)
            model.create_package_rasa(_train_path, "rasa_model", output_path,
                                      new_fingerprint)
            print_success("Your Rasa Core model is trained and saved at '{}'."
                          "".format(output_path))

        return core_model
    else:
        rasa_core.train.do_compare_training(args, args.stories, None)
        return None


def train_nlu(args: argparse.Namespace, train_path: Optional[Text] = None
              ) -> Optional["Interpreter"]:
    import rasa_nlu.train
    from rasa_core.utils import print_success
    from rasa_nlu import config

    args.out = train_path or args.out

    validate(args, [("out", DEFAULT_MODELS_PATH, True),
                    ("config", DEFAULT_CONFIG_PATH),
                    ("nlu", DEFAULT_NLU_DATA_PATH)])

    _train_path = train_path or tempfile.mkdtemp()
    _, nlu_model, _ = rasa_nlu.train.do_train(
        config.load(args.config),
        args.nlu,
        _train_path,
        project="rasa_model",
        fixed_model_name="nlu")

    if not train_path:
        output_path = args.out or create_default_output_path(prefix="nlu-")
        new_fingerprint = model_fingerprint(args.config, nlu_data=args.stories,)
        model.create_package_rasa(_train_path, "rasa_model", output_path,
                                  new_fingerprint)
        print_success("Your Rasa NLU model is trained and saved at '{}'."
                      "".format(output_path))

    return nlu_model
