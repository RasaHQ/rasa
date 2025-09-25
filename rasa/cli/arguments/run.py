import argparse
import os
from typing import Union

from rasa.cli.arguments.default_arguments import (
    add_endpoint_param,
    add_model_param,
    add_remote_storage_param,
    add_skip_validation_flag,
    add_sub_agents_param,
)
from rasa.core import constants
from rasa.env import (
    AUTH_TOKEN_ENV,
    DEFAULT_JWT_METHOD,
    JWT_METHOD_ENV,
    JWT_PRIVATE_KEY_ENV,
    JWT_SECRET_ENV,
)


def set_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments for running Rasa directly using `rasa run`."""
    add_model_param(parser)
    add_server_arguments(parser)
    add_inspect_argument(parser)
    add_skip_validation_flag(parser)
    add_sub_agents_param(parser)


def set_run_action_arguments(parser: argparse.ArgumentParser) -> None:
    """Set arguments for running Rasa SDK."""
    import rasa_sdk.cli.arguments as sdk

    sdk.add_endpoint_arguments(parser)


def add_interface_argument(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],
) -> None:
    """Binds the RASA process to a network interface."""
    parser.add_argument(
        "-i",
        "--interface",
        default=constants.DEFAULT_SERVER_INTERFACE,
        type=str,
        help="Network interface to run the server on.",
    )


# noinspection PyProtectedMember
def add_port_argument(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],
) -> None:
    """Add an argument for port."""
    parser.add_argument(
        "-p",
        "--port",
        default=constants.DEFAULT_SERVER_PORT,
        type=int,
        help="Port to run the server at.",
    )


# noinspection PyProtectedMember
def add_inspect_argument(
    parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup],
) -> None:
    """Add an argument for port."""
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Run development inspector alongside the assistant.",
    )


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for running API endpoint."""
    parser.add_argument(
        "--log-file",
        type=str,
        # Rasa should not log to a file by default, otherwise there will be problems
        # when running on OpenShift
        default=None,
        help="Store logs in specified file.",
    )
    parser.add_argument(
        "--use-syslog", action="store_true", help="Add syslog as a log handler"
    )
    parser.add_argument(
        "--syslog-address",
        type=str,
        default=constants.DEFAULT_SYSLOG_HOST,
        help="Address of the syslog server. --use-sylog flag is required",
    )
    parser.add_argument(
        "--syslog-port",
        type=int,
        default=constants.DEFAULT_SYSLOG_PORT,
        help="Port of the syslog server. --use-sylog flag is required",
    )
    parser.add_argument(
        "--syslog-protocol",
        type=str,
        default=constants.DEFAULT_PROTOCOL,
        help="Protocol used with the syslog server. Can be UDP (default) or TCP ",
    )
    add_endpoint_param(
        parser,
        help_text="Configuration file for the model server and the connectors as a "
        "yml file.",
    )

    add_server_settings_arguments(parser)


def add_server_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the API server.

    Args:
        parser: Argument parser.
    """
    server_arguments = parser.add_argument_group("Server Settings")

    add_interface_argument(server_arguments)
    add_port_argument(server_arguments)

    server_arguments.add_argument(
        "-t",
        "--auth-token",
        type=str,
        default=os.getenv(AUTH_TOKEN_ENV),
        help="Enable token based authentication. Requests need to provide "
        "the token to be accepted.",
    )
    server_arguments.add_argument(
        "--cors",
        nargs="*",
        type=str,
        help="Enable CORS for the passed origin. Use * to whitelist all origins.",
    )
    server_arguments.add_argument(
        "--enable-api",
        action="store_true",
        help="Start the web server API in addition to the input channel.",
    )
    server_arguments.add_argument(
        "--response-timeout",
        default=constants.DEFAULT_RESPONSE_TIMEOUT,
        type=int,
        help="Maximum time a response can take to process (sec).",
    )
    server_arguments.add_argument(
        "--request-timeout",
        default=constants.DEFAULT_REQUEST_TIMEOUT,
        type=int,
        help="Maximum time a request can take to process (sec).",
    )
    add_remote_storage_param(server_arguments)
    server_arguments.add_argument(
        "--ssl-certificate",
        help="Set the SSL Certificate to create a TLS secured server.",
    )
    server_arguments.add_argument(
        "--ssl-keyfile", help="Set the SSL Keyfile to create a TLS secured server."
    )
    server_arguments.add_argument(
        "--ssl-ca-file",
        help="If your SSL certificate needs to be verified, "
        "you can specify the CA file "
        "using this parameter.",
    )
    server_arguments.add_argument(
        "--ssl-password",
        help="If your ssl-keyfile is protected by a password, you can specify it "
        "using this paramer.",
    )
    channel_arguments = parser.add_argument_group("Channels")
    channel_arguments.add_argument(
        "--credentials",
        default=None,
        help="Authentication credentials for the connector as a yml file.",
    )
    channel_arguments.add_argument(
        "--connector", type=str, help="Service to connect to."
    )

    add_jwt_arguments(parser)


def add_jwt_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to JWT authentication.

    Args:
        parser: Argument parser.
    """
    jwt_auth = parser.add_argument_group("JWT Authentication")
    jwt_auth.add_argument(
        "--jwt-secret",
        type=str,
        default=os.getenv(JWT_SECRET_ENV),
        help="Public key for asymmetric JWT methods or shared secret"
        "for symmetric methods. Please also make sure to use "
        "--jwt-method to select the method of the signature, "
        "otherwise this argument will be ignored."
        "Note that this key is meant for securing the HTTP API.",
    )
    jwt_auth.add_argument(
        "--jwt-method",
        type=str,
        default=os.getenv(JWT_METHOD_ENV, DEFAULT_JWT_METHOD),
        help="Method used for the signature of the JWT authentication payload.",
    )
    jwt_auth.add_argument(
        "--jwt-private-key",
        type=str,
        default=os.getenv(JWT_PRIVATE_KEY_ENV),
        help="A private key used for generating web tokens, dependent upon "
        "which hashing algorithm is used. It must be used together with "
        "--jwt-secret for providing the public key.",
    )
