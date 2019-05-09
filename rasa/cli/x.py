import argparse
import datetime
import importlib.util
import logging
import signal
import sys
from multiprocessing import get_context
from typing import List, Text

import rasa.utils.io
import questionary

import rasa.cli.run
import rasa.core.utils
from rasa.cli.utils import print_success, get_validated_path
from rasa.cli.arguments.arguments import add_logging_option_arguments

from rasa.constants import (
    GLOBAL_USER_CONFIG_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_LOG_LEVEL,
)
from rasa.utils.common import read_global_config_value, write_global_config_value

logger = logging.getLogger(__name__)


# noinspection PyProtectedMember
def add_subparser(
    subparsers: argparse._SubParsersAction, parents: List[argparse.ArgumentParser]
):
    x_parser_args = {
        "parents": parents,
        "conflict_handler": "resolve",
        "formatter_class": argparse.ArgumentDefaultsHelpFormatter,
    }

    if is_rasa_x_installed():
        # we'll only show the help msg for the command if Rasa X is actually installed
        x_parser_args["help"] = "Start Rasa X and the Interface"

    shell_parser = subparsers.add_parser("x", **x_parser_args)

    shell_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatic yes or default options to prompts and oppressed warnings",
    )

    shell_parser.add_argument(
        "--production",
        action="store_true",
        help="Run Rasa X in a production environment",
    )

    shell_parser.add_argument("--auth-token", type=str, help="Rasa API auth token")

    shell_parser.add_argument(
        "--nlg",
        type=str,
        default="http://localhost:5002/api/nlg",
        help="Rasa NLG endpoint",
    )

    shell_parser.add_argument(
        "--model-endpoint-url",
        type=str,
        default="http://localhost:5002/api/projects/default/models/tags/production",
        help="Rasa model endpoint URL",
    )

    shell_parser.add_argument(
        "--project-path",
        type=str,
        default=".",
        help="Path to the Rasa project directory",
    )

    shell_parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help=(
            "Path to the directory containing Rasa NLU training data "
            "and Rasa Core stories"
        ),
    )

    rasa.cli.run.add_run_arguments(shell_parser)
    add_logging_option_arguments(shell_parser)

    shell_parser.set_defaults(func=rasa_x)


def _event_service():
    """Start the event service."""
    # noinspection PyUnresolvedReferences
    from rasax.community.services.event_service import main

    main()


def start_event_service():
    """Run the event service in a separate process."""

    ctx = get_context("spawn")
    p = ctx.Process(target=_event_service)
    p.start()


def is_metrics_collection_enabled(args: argparse.Namespace) -> bool:
    """Make sure the user consents to any metrics collection."""

    try:
        allow_metrics = read_global_config_value("metrics", unavailable_ok=False)
        return allow_metrics.get("enabled", False)
    except ValueError:
        pass  # swallow the error and ask the user

    allow_metrics = (
        questionary.confirm(
            "Rasa will track a minimal amount of anonymized usage information "
            "(like how often you use the 'train' button) to help us improve Rasa X. "
            "None of your training data or conversations will ever be sent to Rasa. "
            "Are you OK with Rasa collecting anonymized usage data?"
        )
        .skip_if(args.no_prompt, default=True)
        .ask()
    )

    print_success(
        "Your decision has been stored into {}. " "".format(GLOBAL_USER_CONFIG_PATH)
    )

    if not args.no_prompt:
        date = datetime.datetime.now()
        write_global_config_value("metrics", {"enabled": allow_metrics, "date": date})

    return allow_metrics


def _core_service(args: argparse.Namespace, endpoints: "AvailableEndpoints" = None):
    """Starts the Rasa Core application."""
    from rasa.core.run import serve_application
    from rasa.nlu.utils import configure_colored_logging

    configure_colored_logging(args.loglevel)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)

    args.credentials = get_validated_path(
        args.credentials, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )

    if endpoints is None:
        args.endpoints = get_validated_path(
            args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
        )
        from rasa.core.utils import AvailableEndpoints

        endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

    serve_application(
        endpoints=endpoints,
        port=args.port,
        credentials=args.credentials,
        cors=args.cors,
        auth_token=args.auth_token,
        enable_api=True,
        jwt_secret=args.jwt_secret,
        jwt_method=args.jwt_method,
    )


def start_core_for_local_platform(args: argparse.Namespace, rasa_x_token: Text):
    """Starts the Rasa X API with Rasa as a background process."""

    from rasa.core.utils import AvailableEndpoints
    from rasa.utils.endpoints import EndpointConfig

    endpoints = AvailableEndpoints(
        model=EndpointConfig(
            args.model_endpoint_url, token=rasa_x_token, wait_time_between_pulls=2
        ),
        nlg=EndpointConfig(args.nlg, token=rasa_x_token),
        tracker_store=EndpointConfig(type="sql", db="tracker.db"),
    )

    vars(args).update(
        dict(
            nlu_model=None,
            channel="rasa",
            credentials="credentials.yml",
            cors="*",
            auth_token=args.auth_token,
            enable_api=True,
            endpoints=endpoints,
        )
    )

    ctx = get_context("spawn")
    p = ctx.Process(target=_core_service, args=(args, endpoints))
    p.start()


def is_rasa_x_installed():
    """Check if Rasa X is installed."""

    # we could also do something like checking if `import rasa_platform` works,
    # the issue with that is that it actually does import the package and this
    # takes some time that we don't want to spend when booting the CLI
    return importlib.util.find_spec("rasax") is not None


def generate_rasa_x_token(length=16):
    """Generate a hexadecimal secret token used to access the Rasa X API.

    A new token is generated on every `rasa x` command.
    """

    from secrets import token_hex

    return token_hex(length)


def rasa_x(args: argparse.Namespace):
    from rasa.cli.utils import print_success, print_error, signal_handler
    from rasa.core.utils import configure_file_logging
    from rasa.utils.io import configure_colored_logging

    signal.signal(signal.SIGINT, signal_handler)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.ERROR)

    if not args.loglevel == logging.DEBUG:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)
        logging.getLogger("apscheduler").setLevel(logging.ERROR)
        logging.getLogger("rasa").setLevel(logging.WARNING)
        logging.getLogger("sanic.root").setLevel(logging.ERROR)

    args.log_level = args.loglevel or DEFAULT_LOG_LEVEL
    configure_colored_logging(args.log_level)
    configure_file_logging(args.log_level, args.log_file)

    metrics = is_metrics_collection_enabled(args)

    if args.production:
        print_success("Starting Rasa X in production mode... 🚀")
        _core_service(args)
    else:
        print_success("Starting Rasa X in local mode... 🚀")
        if not is_rasa_x_installed():
            print_error(
                "Rasa X is not installed. The `rasa x` "
                "command requires an installation of Rasa X."
            )
            sys.exit(1)

        # noinspection PyUnresolvedReferences
        from rasax.community.api.local import main_local

        start_event_service()

        rasa_x_token = generate_rasa_x_token()

        start_core_for_local_platform(args, rasa_x_token=rasa_x_token)

        main_local(
            args.project_path, args.data_path, token=rasa_x_token, metrics=metrics
        )
