import argparse
import logging
import os
from typing import List, Text

from rasa.api import run as rasa_run
from rasa.cli import SubParsersAction
from rasa.cli.arguments import run as arguments
from rasa.cli.arguments.default_arguments import SkipYamlValidation
from rasa.core.config.configuration import (
    Configuration,
)
from rasa.exceptions import ModelNotFound
from rasa.shared.constants import (
    DEFAULT_ACTIONS_PATH,
    DEFAULT_MODELS_PATH,
    DOCS_BASE_URL,
)
from rasa.shared.core.domain import Domain
from rasa.shared.utils.cli import print_error

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all run parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    run_parser = subparsers.add_parser(
        "run",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Starts a Rasa server with your trained model.",
    )
    run_parser.set_defaults(func=run)

    run_subparsers = run_parser.add_subparsers()
    sdk_subparser = run_subparsers.add_parser(
        "actions",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Runs the action server.",
    )
    sdk_subparser.set_defaults(func=run_actions)

    arguments.set_run_arguments(run_parser)
    arguments.set_run_action_arguments(sdk_subparser)


def run_actions(args: argparse.Namespace) -> None:
    import rasa_sdk.__main__ as sdk

    args.actions = args.actions or DEFAULT_ACTIONS_PATH

    sdk.main_from_args(args)


def _validate_model_path(model_path: Text, parameter: Text, default: Text) -> Text:
    if model_path is not None and not os.path.exists(model_path):
        raise ModelNotFound(
            f"The provided model path '{model_path}' could not be found. "
            "Provide an existing model path."
        )

    if model_path is None:
        logger.debug(
            f"Parameter '{parameter}' not set. "
            "Using default location '{default}' instead."
        )
        os.makedirs(default, exist_ok=True)
        model_path = default

    return model_path


def run(args: argparse.Namespace) -> None:
    """Entrypoint for `rasa run`.

    Args:
        args: The CLI arguments.
    """
    Configuration.initialise_endpoints(
        args.endpoints,
    ).initialise_sub_agents(args.sub_agents).initialise_credentials(
        args.credentials,
    )

    if SkipYamlValidation.DOMAIN.value in args.skip_yaml_validation:
        Domain.validate_yaml = False
    else:
        Domain.validate_yaml = True

    if args.enable_api:
        if not args.remote_storage:
            args.model = _validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)
        rasa_run(**vars(args))
        return

    # if the API is not enable you cannot start without a model
    # make sure either a model server, a remote storage, or a local model is
    # configured

    import rasa.model

    # start server if remote storage is configured
    if args.remote_storage is not None:
        rasa_run(**vars(args))
        return

    # start server if model server is configured
    endpoints = Configuration.get_instance().endpoints
    model_server = endpoints.model if endpoints and endpoints.model else None
    if model_server is not None:
        rasa_run(**vars(args))
        return

    # start server if local model found
    args.model = _validate_model_path(args.model, "model", DEFAULT_MODELS_PATH)
    local_model_set = True
    try:
        rasa.model.get_local_model(args.model)
    except ModelNotFound:
        local_model_set = False

    if local_model_set:
        rasa_run(**vars(args))
        return

    print_error(
        f"No model found. You have three options to provide a model:\n"
        f"1. Configure a model server in the endpoint configuration and provide "
        f"the configuration via '--endpoints'.\n"
        f"2. Specify a remote storage via '--remote-storage' to load the model "
        f"from.\n"
        f"3. Train a model before running the server using `rasa train` and "
        f"use '--model' to provide the model path.\n"
        f"For more information check {DOCS_BASE_URL}/model-storage."
    )
