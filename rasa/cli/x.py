import argparse
import asyncio
import logging
import signal
from pathlib import Path
from typing import Iterable, List, Optional, Text, Tuple, Union

import aiohttp
import ruamel.yaml as yaml

import rasa.cli.utils
import rasa.shared.utils.cli
import rasa.shared.utils.io
import rasa.utils.common
import rasa.utils.io
from rasa.cli import SubParsersAction
from rasa.cli.arguments import x as arguments
from rasa.cli.validation.config_path_validation import get_validated_path
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.configuration import Configuration
from rasa.shared.constants import (
    DEFAULT_CREDENTIALS_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.shared.utils.yaml import read_config_file

logger = logging.getLogger(__name__)


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

    x_parser_args["help"] = (
        "Run a Rasa server in a mode that enables connecting "
        "to Rasa Enterprise as the config endpoint."
    )

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
    rasa.utils.common.configure_logging_and_warnings(args.loglevel)
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
    if credentials_path:
        credentials_path = str(
            get_validated_path(
                credentials_path, "credentials", DEFAULT_CREDENTIALS_PATH, True
            )
        )
        credentials = read_config_file(credentials_path)
    else:
        credentials = {}

    # this makes sure the Rasa X is properly configured no matter what
    if rasa_x_url:
        credentials["rasa"] = {"url": rasa_x_url}
    dumped_credentials = yaml.dump(credentials, default_flow_style=False)
    tmp_credentials = rasa.utils.io.create_temporary_file(dumped_credentials, "yml")

    return tmp_credentials


def rasa_x(args: argparse.Namespace) -> None:
    """Run Rasa with the `x` subcommand."""
    from rasa.cli.utils import signal_handler

    signal.signal(signal.SIGINT, signal_handler)

    if args.production:
        run_in_enterprise_connection_mode(args)
    else:
        rasa.shared.utils.io.raise_warning(
            "Running Rasa X in local mode is no longer supported as Rasa has "
            "stopped supporting the Community Edition (free version) of ‘Rasa X’."
            "For more information please see "
            "https://rasa.com/blog/rasa-x-community-edition-changes/",
            UserWarning,
        )
        exit()


async def _pull_runtime_config_from_server(
    config_endpoint: Optional[Text],
    attempts: int = 60,
    wait_time_between_pulls: float = 5,
    keys: Iterable[Text] = ("endpoints", "credentials"),
) -> List[Text]:
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
        "Could not fetch runtime config from server at '{}'. Exiting.".format(
            config_endpoint
        )
    )


def run_in_enterprise_connection_mode(args: argparse.Namespace) -> None:
    """Run Rasa in a mode that enables using Rasa X as the config endpoint."""
    from rasa.shared.utils.cli import print_success

    print_success("Starting a Rasa server in Rasa Enterprise connection mode... 🚀")

    credentials_path, endpoints_path = _get_credentials_and_endpoints_paths(args)
    endpoints = Configuration.initialise_endpoints(
        endpoints_path=Path(endpoints_path)
    ).endpoints

    _rasa_service(args, endpoints, None, credentials_path)


def _get_credentials_and_endpoints_paths(
    args: argparse.Namespace,
) -> Tuple[Optional[Text], Optional[Text]]:
    config_endpoint = args.config_endpoint
    endpoints_config_path: Optional[Union[Path, Text]]

    if config_endpoint:
        endpoints_config_path, credentials_path = asyncio.run(
            _pull_runtime_config_from_server(config_endpoint)
        )
    else:
        endpoints_config_path = get_validated_path(
            args.endpoints, "endpoints", DEFAULT_ENDPOINTS_PATH, True
        )
        credentials_path = None

    return (
        credentials_path,
        str(endpoints_config_path) if endpoints_config_path else None,
    )
