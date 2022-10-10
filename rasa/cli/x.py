import argparse
import asyncio
import importlib.util
import logging
from multiprocessing import get_context, Process
import os
import signal
import sys
import traceback
from typing import Iterable, List, Optional, Text, Tuple

import aiohttp
from rasa.exceptions import MissingDependencyException
import ruamel.yaml as yaml

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import x as arguments
import rasa.cli.utils
from rasa.constants import (
    DEFAULT_LOG_LEVEL_RASA_X,
    DEFAULT_RASA_PORT,
    DEFAULT_RASA_X_PORT,
)
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DOCS_BASE_URL_RASA_X,
)
from rasa.core.utils import AvailableEndpoints
from rasa.shared.exceptions import RasaXTermsError
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.utils.common
from rasa.utils.endpoints import EndpointConfig
import rasa.utils.io

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_DB = "events.db"


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all rasa x parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
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
    endpoints: AvailableEndpoints,
    rasa_x_url: Optional[Text] = None,
    credentials_path: Optional[Text] = None,
) -> None:
    """Starts the Rasa application."""
    from rasa.core.run import serve_application

    # needs separate logging configuration as it is started in its own process
    rasa.utils.common.set_log_level(args.loglevel)
    rasa.utils.io.configure_colored_logging(args.loglevel)

    if not credentials_path:
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
        ssl_certificate=args.ssl_certificate,
        ssl_keyfile=args.ssl_keyfile,
        ssl_ca_file=args.ssl_ca_file,
        ssl_password=args.ssl_password,
    )


def _prepare_credentials_for_rasa_x(
    credentials_path: Optional[Text], rasa_x_url: Optional[Text] = None
) -> Text:
    credentials_path = rasa.cli.utils.get_validated_path(
        credentials_path, "credentials", DEFAULT_CREDENTIALS_PATH, True
    )
    if credentials_path:
        credentials = rasa.shared.utils.io.read_config_file(credentials_path)
    else:
        credentials = {}

    # this makes sure the Rasa X is properly configured no matter what
    if rasa_x_url:
        credentials["rasa"] = {"url": rasa_x_url}
    dumped_credentials = yaml.dump(credentials, default_flow_style=False)
    tmp_credentials = rasa.utils.io.create_temporary_file(dumped_credentials, "yml")

    return tmp_credentials


def _overwrite_endpoints_for_local_x(
    endpoints: AvailableEndpoints, rasa_x_token: Text, rasa_x_url: Text
) -> None:
    endpoints.model = _get_model_endpoint(endpoints.model, rasa_x_token, rasa_x_url)
    endpoints.event_broker = _get_event_broker_endpoint(endpoints.event_broker)


def _get_model_endpoint(
    model_endpoint: Optional[EndpointConfig], rasa_x_token: Text, rasa_x_url: Text
) -> EndpointConfig:
    # If you change that, please run a test with Rasa X and speak to the bot
    default_rasax_model_server_url = f"{rasa_x_url}/models/tags/production"

    model_endpoint = model_endpoint or EndpointConfig()

    # Checking if endpoint.yml has existing url, if so give
    # warning we are overwriting the endpoint.yml file.
    custom_url = model_endpoint.url

    if custom_url and custom_url != default_rasax_model_server_url:
        logger.info(
            f"Ignoring url '{custom_url}' from 'endpoints.yml' and using "
            f"'{default_rasax_model_server_url}' instead."
        )

    custom_wait_time_pulls = model_endpoint.kwargs.get("wait_time_between_pulls")
    return EndpointConfig(
        default_rasax_model_server_url,
        token=rasa_x_token,
        wait_time_between_pulls=custom_wait_time_pulls or 2,
    )


def _get_event_broker_endpoint(
    event_broker_endpoint: Optional[EndpointConfig],
) -> EndpointConfig:
    import questionary

    default_event_broker_endpoint = EndpointConfig(
        type="sql", dialect="sqlite", db=DEFAULT_EVENTS_DB
    )
    if not event_broker_endpoint:
        return default_event_broker_endpoint
    elif not _is_correct_event_broker(event_broker_endpoint):
        rasa.shared.utils.cli.print_error(
            f"Rasa X currently only supports a SQLite event broker with path "
            f"'{DEFAULT_EVENTS_DB}' when running locally. You can deploy Rasa X "
            f"with Docker ({DOCS_BASE_URL_RASA_X}/installation-and-setup/"
            f"docker-compose-quick-install/) if you want to use other event broker "
            f"configurations."
        )
        continue_with_default_event_broker = questionary.confirm(
            "Do you want to continue with the default SQLite event broker?"
        ).ask()

        if not continue_with_default_event_broker:
            sys.exit(0)

        return default_event_broker_endpoint
    else:
        return event_broker_endpoint


def _is_correct_event_broker(event_broker: EndpointConfig) -> bool:
    return all(
        [
            event_broker.type == "sql",
            event_broker.kwargs.get("dialect", "").lower() == "sqlite",
            event_broker.kwargs.get("db") == DEFAULT_EVENTS_DB,
        ]
    )


def start_rasa_for_local_rasa_x(
    args: argparse.Namespace, rasa_x_token: Text
) -> Process:
    """Starts the Rasa X API with Rasa as a background process."""
    credentials_path, endpoints_path = _get_credentials_and_endpoints_paths(args)
    endpoints = AvailableEndpoints.read_endpoints(endpoints_path)

    rasa_x_url = f"http://localhost:{args.rasa_x_port}/api"
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
    p = ctx.Process(
        target=_rasa_service,
        args=(args, endpoints, rasa_x_url, credentials_path),
        daemon=True,
    )
    p.start()
    return p


def is_rasa_x_installed() -> bool:
    """Check if Rasa X is installed."""

    # we could also do something like checking if `import rasax` works,
    # the issue with that is that it actually does import the package and this
    # takes some time that we don't want to spend when booting the CLI
    return importlib.util.find_spec("rasax") is not None


def generate_rasa_x_token(length: int = 16) -> Text:
    """Generate a hexadecimal secret token used to access the Rasa X API.

    A new token is generated on every `rasa x` command.
    """

    from secrets import token_hex

    return token_hex(length)


def _configure_logging(args: argparse.Namespace) -> None:
    from rasa.core.utils import configure_file_logging
    from rasa.utils.common import set_log_level

    log_level = args.loglevel or DEFAULT_LOG_LEVEL_RASA_X

    if isinstance(log_level, str):
        log_level = logging.getLevelName(log_level)

    logging.basicConfig(level=log_level)
    rasa.utils.io.configure_colored_logging(args.loglevel)

    set_log_level(log_level)
    configure_file_logging(logging.root, args.log_file)

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("engineio").setLevel(logging.WARNING)
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("socketio").setLevel(logging.ERROR)

    if not log_level == logging.DEBUG:
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)


def is_rasa_project_setup(args: argparse.Namespace, project_path: Text) -> bool:
    """Checks if `project_path` contains a valid Rasa Open Source project.

    Args:
        args: Command-line arguments.
        project_path: Path to the possible Rasa Open Source project.

    Returns:
        `True` if `project_path` is a valid Rasa Open Source project, `False` otherwise.
    """
    config_path = _get_config_path(args)
    domain_path = _get_domain_path(args)

    mandatory_files = [config_path, domain_path]

    for f in mandatory_files:
        if not os.path.exists(os.path.join(project_path, f)):
            return False

    return True


def _validate_rasa_x_start(args: argparse.Namespace, project_path: Text) -> None:
    if not is_rasa_x_installed():
        rasa.shared.utils.cli.print_error_and_exit(
            "Rasa X is not installed. The `rasa x` "
            "command requires an installation of Rasa X. "
            "Instructions on how to install Rasa X can be found here: "
            "https://rasa.com/docs/rasa-x/."
        )

    if args.port == args.rasa_x_port:
        rasa.shared.utils.cli.print_error_and_exit(
            "The port for Rasa X '{}' and the port of the Rasa server '{}' are the "
            "same. We need two different ports, one to run Rasa X (e.g. delivering the "
            "UI) and another one to run a normal Rasa server.\nPlease specify two "
            "different ports using the arguments '--port' and '--rasa-x-port'.".format(
                args.rasa_x_port, args.port
            )
        )

    if not is_rasa_project_setup(args, project_path):
        rasa.shared.utils.cli.print_error_and_exit(
            "This directory is not a valid Rasa project. Use 'rasa init' "
            "to create a new Rasa project or switch to a valid Rasa project "
            "directory (see "
            "https://rasa.com/docs/rasa/command-line-interface#rasa-init)."
        )

    domain_path = _get_domain_path(args)
    _validate_domain(os.path.join(project_path, domain_path))

    if args.data and not os.path.exists(args.data):
        rasa.shared.utils.cli.print_warning(
            "The provided data path ('{}') does not exists. Rasa X will start "
            "without any training data.".format(args.data)
        )


def _validate_domain(domain_path: Text) -> None:
    from rasa.shared.core.domain import Domain, InvalidDomain

    try:
        Domain.load(domain_path)
    except InvalidDomain as e:
        rasa.shared.utils.cli.print_error_and_exit(
            "The provided domain file could not be loaded. " "Error: {}".format(e)
        )


def rasa_x(args: argparse.Namespace) -> None:
    from rasa.cli.utils import signal_handler

    signal.signal(signal.SIGINT, signal_handler)

    _configure_logging(args)

    if args.production:
        run_in_production(args)
    else:
        run_locally(args)


async def _pull_runtime_config_from_server(
    config_endpoint: Optional[Text],
    attempts: int = 60,
    wait_time_between_pulls: float = 5,
    keys: Iterable[Text] = ("endpoints", "credentials"),
) -> Optional[List[Text]]:
    """Pull runtime config from `config_endpoint`.

    Returns a list of paths to yaml dumps, each containing the contents of one of
    `keys`.
    """

    while attempts:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(config_endpoint) as resp:
                    if resp.status == 200:
                        rjs = await resp.json()
                        try:
                            return [
                                rasa.utils.io.create_temporary_file(rjs[k])
                                for k in keys
                            ]
                        except KeyError as e:
                            rasa.shared.utils.cli.print_error_and_exit(
                                "Failed to find key '{}' in runtime config. "
                                "Exiting.".format(e)
                            )
                    else:
                        logger.debug(
                            "Failed to get a proper response from remote "
                            "server. Status Code: {}. Response: '{}'"
                            "".format(resp.status, await resp.text())
                        )
        except aiohttp.ClientError as e:
            logger.debug(f"Failed to connect to server. Retrying. {e}")

        await asyncio.sleep(wait_time_between_pulls)
        attempts -= 1

    rasa.shared.utils.cli.print_error_and_exit(
        "Could not fetch runtime config from server at '{}'. "
        "Exiting.".format(config_endpoint)
    )


def run_in_production(args: argparse.Namespace) -> None:
    from rasa.shared.utils.cli import print_success

    print_success("Starting Rasa X in production mode... 🚀")

    credentials_path, endpoints_path = _get_credentials_and_endpoints_paths(args)
    endpoints = AvailableEndpoints.read_endpoints(endpoints_path)

    _rasa_service(args, endpoints, None, credentials_path)


def _get_config_path(args: argparse.Namespace,) -> Optional[Text]:
    config_path = rasa.cli.utils.get_validated_path(
        args.config, "config", DEFAULT_CONFIG_PATH
    )

    return config_path


def _get_domain_path(args: argparse.Namespace,) -> Optional[Text]:
    domain_path = rasa.cli.utils.get_validated_path(
        args.domain, "domain", DEFAULT_DOMAIN_PATH
    )

    return domain_path


def _get_credentials_and_endpoints_paths(
    args: argparse.Namespace,
) -> Tuple[Optional[Text], Optional[Text]]:
    config_endpoint = args.config_endpoint
    if config_endpoint:
        endpoints_config_path, credentials_path = rasa.utils.common.run_in_loop(
            _pull_runtime_config_from_server(config_endpoint)
        )

    else:
        endpoints_config_path = rasa.cli.utils.get_validated_path(
            args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
        )
        credentials_path = None

    return credentials_path, endpoints_config_path


def _prevent_failure_if_git_is_not_available() -> None:
    """Rasa X uses the `git` package, which will fail to import if git is not available.

    Git isn't needed locally, which means we can silence this error to allow
    users to use local mode even if git is not available on their machine.
    Fixes regression https://github.com/RasaHQ/rasa/issues/7140
    """
    if os.environ.get("GIT_PYTHON_REFRESH") is None:
        os.environ["GIT_PYTHON_REFRESH"] = "quiet"


def run_locally(args: argparse.Namespace) -> None:
    """Run a Rasa X instance locally.

    Args:
        args: commandline arguments
    """
    _prevent_failure_if_git_is_not_available()

    try:
        # noinspection PyUnresolvedReferences
        from rasax.community import local
    except ModuleNotFoundError:
        raise MissingDependencyException(
            f"Rasa X does not seem to be installed, but it is needed for this "
            f"CLI command. You can find more information on how to install Rasa X "
            f"in local mode in the documentation: "
            f"{DOCS_BASE_URL_RASA_X}/installation-and-setup/install/local-mode"
        )

    args.rasa_x_port = args.rasa_x_port or DEFAULT_RASA_X_PORT
    args.port = args.port or DEFAULT_RASA_PORT

    project_path = "."

    _validate_rasa_x_start(args, project_path)

    rasa_x_token = generate_rasa_x_token()
    process = start_rasa_for_local_rasa_x(args, rasa_x_token=rasa_x_token)

    config_path = _get_config_path(args)
    domain_path = _get_domain_path(args)

    telemetry.track_rasa_x_local()

    # noinspection PyBroadException
    try:
        try:
            local.main(
                args,
                project_path,
                args.data,
                token=rasa_x_token,
                config_path=config_path,
                domain_path=domain_path,
            )
        except TypeError as e:
            if "domain_path" in str(e):
                # backwards compatibility with Rasa X versions < 0.35.0
                # fixes regression https://github.com/RasaHQ/rasa/issues/7592
                local.main(
                    args,
                    project_path,
                    args.data,
                    token=rasa_x_token,
                    config_path=config_path,
                )
            else:
                raise
    except RasaXTermsError:
        # User didn't accept the Rasa X terms.
        pass
    except Exception:
        print(traceback.format_exc())
        rasa.shared.utils.cli.print_error(
            "Sorry, something went wrong (see error above). Make sure to start "
            "Rasa X with valid data and valid domain and config files. Please, "
            "also check any warnings that popped up.\nIf you need help fixing "
            "the issue visit our forum: https://forum.rasa.com/."
        )
    finally:
        process.terminate()
