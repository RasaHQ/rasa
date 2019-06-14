import argparse
import importlib.util
import logging
import signal
import sys
import os
import traceback
from multiprocessing import get_context
import typing
from typing import List, Text, Optional

import ruamel.yaml as yaml

from rasa.cli.utils import get_validated_path, print_warning, print_error
from rasa.cli.arguments import x as arguments

from rasa.constants import (
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOG_LEVEL_RASA_X,
    DEFAULT_RASA_X_PORT,
    DEFAULT_RASA_PORT,
)
import rasa.utils.io as io_utils
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

DEFAULT_TRACKER_DB = "tracker.db"

if typing.TYPE_CHECKING:
    from rasa.core.utils import AvailableEndpoints


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
        x_parser_args["help"] = "Starts the Rasa X interface."

    shell_parser = subparsers.add_parser("x", **x_parser_args)
    shell_parser.set_defaults(func=rasa_x)

    arguments.set_x_arguments(shell_parser)


def _rasa_service(
    args: argparse.Namespace,
    endpoints: "AvailableEndpoints",
    rasa_x_url: Optional[Text] = None,
):
    """Starts the Rasa application."""
    from rasa.core.run import serve_application

    # needs separate logging configuration as it is started in its own process
    logging.basicConfig(level=args.loglevel)
    io_utils.configure_colored_logging(args.loglevel)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    credentials_path = _prepare_credentials_for_rasa_x(
        args.credentials, rasa_x_url=rasa_x_url
    )

    serve_application(
        endpoints=endpoints,
        port=args.port,
        credentials=credentials_path,
        cors=args.cors,
        auth_token=args.auth_token,
        enable_api=True,
        jwt_secret=args.jwt_secret,
        jwt_method=args.jwt_method,
    )


def _prepare_credentials_for_rasa_x(
    credentials_path: Optional[Text], rasa_x_url: Optional[Text] = None
) -> Text:
    credentials_path = get_validated_path(
        credentials_path, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )
    if credentials_path:
        credentials = io_utils.read_config_file(credentials_path)
    else:
        credentials = {}

    # this makes sure the Rasa X is properly configured no matter what
    if rasa_x_url:
        credentials["rasa"] = {"url": rasa_x_url}
    dumped_credentials = yaml.dump(credentials, default_flow_style=False)
    tmp_credentials = io_utils.create_temporary_file(dumped_credentials, "yml")

    return tmp_credentials


def _overwrite_endpoints_for_local_x(
    endpoints: "AvailableEndpoints", rasa_x_token: Text, rasa_x_url: Text
):
    from rasa.utils.endpoints import EndpointConfig
    import questionary

    endpoints.model = EndpointConfig(
        "{}/projects/default/models/tags/production".format(rasa_x_url),
        token=rasa_x_token,
        wait_time_between_pulls=2,
    )

    overwrite_existing_tracker_store = False
    if endpoints.tracker_store and not _is_correct_tracker_store(
        endpoints.tracker_store
    ):
        print_error(
            "Rasa X currently only supports a SQLite tracker store with path '{}' "
            "when running locally. You can deploy Rasa X with Docker "
            "(https://rasa.com/docs/rasa-x/deploy/) if you want to use "
            "other tracker store configurations.".format(DEFAULT_TRACKER_DB)
        )
        overwrite_existing_tracker_store = questionary.confirm(
            "Do you want to continue with the default SQLite tracker store?"
        ).ask()

        if not overwrite_existing_tracker_store:
            exit(0)

    if not endpoints.tracker_store or overwrite_existing_tracker_store:
        endpoints.tracker_store = EndpointConfig(type="sql", db=DEFAULT_TRACKER_DB)


def _is_correct_tracker_store(tracker_endpoint: EndpointConfig) -> bool:
    return (
        tracker_endpoint.type == "sql"
        and tracker_endpoint.kwargs.get("dialect", "").lower() == "sqlite"
        and tracker_endpoint.kwargs.get("db") == DEFAULT_TRACKER_DB
    )


def start_rasa_for_local_rasa_x(args: argparse.Namespace, rasa_x_token: Text):
    """Starts the Rasa X API with Rasa as a background process."""

    from rasa.core.utils import AvailableEndpoints

    args.endpoints = get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )

    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)

    rasa_x_url = "http://localhost:{}/api".format(args.rasa_x_port)
    _overwrite_endpoints_for_local_x(endpoints, rasa_x_token, rasa_x_url)

    vars(args).update(
        dict(
            nlu_model=None,
            cors="*",
            auth_token=args.auth_token,
            enable_api=True,
            endpoints=endpoints,
        )
    )

    ctx = get_context("spawn")
    p = ctx.Process(target=_rasa_service, args=(args, endpoints, rasa_x_url))
    p.daemon = True
    p.start()
    return p


def is_rasa_x_installed():
    """Check if Rasa X is installed."""

    # we could also do something like checking if `import rasax` works,
    # the issue with that is that it actually does import the package and this
    # takes some time that we don't want to spend when booting the CLI
    return importlib.util.find_spec("rasax") is not None


def generate_rasa_x_token(length: int = 16):
    """Generate a hexadecimal secret token used to access the Rasa X API.

    A new token is generated on every `rasa x` command.
    """

    from secrets import token_hex

    return token_hex(length)


def _configure_logging(args: argparse.Namespace):
    from rasa.core.utils import configure_file_logging
    from rasa.utils.common import set_log_level

    log_level = args.loglevel or DEFAULT_LOG_LEVEL_RASA_X

    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)

    logging.basicConfig(level=log_level)
    io_utils.configure_colored_logging(args.loglevel)

    set_log_level(log_level)
    configure_file_logging(args.log_file)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.ERROR)

    if not log_level == logging.DEBUG:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)


def is_rasa_project_setup(project_path: Text):
    mandatory_files = [DEFAULT_CONFIG_PATH, DEFAULT_DOMAIN_PATH]

    for f in mandatory_files:
        if not os.path.exists(os.path.join(project_path, f)):
            return False

    return True


def _validate_rasa_x_start(args: argparse.Namespace, project_path: Text):
    if not is_rasa_x_installed():
        print_error(
            "Rasa X is not installed. The `rasa x` "
            "command requires an installation of Rasa X. "
            "Instructions on how to install Rasa X can be found here: "
            "https://rasa.com/docs/rasa-x/installation-and-setup/."
        )
        sys.exit(1)

    if args.port == args.rasa_x_port:
        print_error(
            "The port for Rasa X '{}' and the port of the Rasa server '{}' are the "
            "same. We need two different ports, one to run Rasa X (e.g. delivering the "
            "UI) and another one to run a normal Rasa server.\nPlease specify two "
            "different ports using the arguments '--port' and '--rasa-x-port'.".format(
                args.rasa_x_port, args.port
            )
        )
        sys.exit(1)

    if not is_rasa_project_setup(project_path):
        print_error(
            "This directory is not a valid Rasa project. Use 'rasa init' "
            "to create a new Rasa project or switch to a valid Rasa project "
            "directory (see http://rasa.com/docs/rasa/user-guide/rasa-tutorial/#create-a-new-project)."
        )
        sys.exit(1)

    _validate_domain(os.path.join(project_path, DEFAULT_DOMAIN_PATH))

    if args.data and not os.path.exists(args.data):
        print_warning(
            "The provided data path ('{}') does not exists. Rasa X will start "
            "without any training data.".format(args.data)
        )


def _validate_domain(domain_path: Text):
    from rasa.core.domain import Domain, InvalidDomain

    try:
        Domain.load(domain_path)
    except InvalidDomain as e:
        print_error("The provided domain file could not be loaded. Error: {}".format(e))
        sys.exit(1)


def rasa_x(args: argparse.Namespace):
    from rasa.cli.utils import signal_handler

    signal.signal(signal.SIGINT, signal_handler)

    _configure_logging(args)

    if args.production:
        run_in_production(args)
    else:
        run_locally(args)


def run_in_production(args: argparse.Namespace):
    from rasa.cli.utils import print_success
    from rasa.core.utils import AvailableEndpoints

    print_success("Starting Rasa X in production mode... 🚀")
    args.endpoints = get_validated_path(
        args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
    )
    endpoints = AvailableEndpoints.read_endpoints(args.endpoints)
    _rasa_service(args, endpoints)


def run_locally(args: argparse.Namespace):
    # noinspection PyUnresolvedReferences
    from rasax.community import local  # pytype: disable=import-error

    args.rasa_x_port = args.rasa_x_port or DEFAULT_RASA_X_PORT
    args.port = args.port or DEFAULT_RASA_PORT

    project_path = "."

    _validate_rasa_x_start(args, project_path)

    local.check_license_and_metrics(args)
    rasa_x_token = generate_rasa_x_token()
    process = start_rasa_for_local_rasa_x(args, rasa_x_token=rasa_x_token)

    try:
        local.main(args, project_path, args.data, token=rasa_x_token)
    except Exception:
        print (traceback.format_exc())
        print_error(
            "Sorry, something went wrong (see error above). Make sure to start "
            "Rasa X with valid data and valid domain and config files. Please, "
            "also check any warnings that popped up.\nIf you need help fixing "
            "the issue visit our forum: https://forum.rasa.com/."
        )
    finally:
        process.terminate()
