import argparse

from rasa.cli.arguments.default_arguments import add_model_param
from rasa.core import constants


def set_run_arguments(parser: argparse.ArgumentParser):
    add_server_arguments(parser)
    add_model_param(parser)


def set_run_action_arguments(parser: argparse.ArgumentParser):
    import rasa_sdk.cli.arguments as sdk

    sdk.add_endpoint_arguments(parser)

    parser.add_argument(
        "--actions",
        type=str,
        default="actions",
        help="Name of action package to be loaded.",
    )


def add_server_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--log-file",
        type=str,
        default="rasa_core.log",
        help="Store logs in specified file.",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Configuration file for the model server and the connectors as a yml file.",
    )

    server_arguments = parser.add_argument_group("Server Settings")
    server_arguments.add_argument(
        "-p",
        "--port",
        default=constants.DEFAULT_SERVER_PORT,
        type=int,
        help="Port to run the server at.",
    )
    server_arguments.add_argument(
        "-t",
        "--auth-token",
        type=str,
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
        help="Start the web server api in addition to the input channel.",
    )
    server_arguments.add_argument(
        "--remote-storage",
        help="Set the remote location where models are stored. "
        "E.g. on AWS. If nothing is configured, the "
        "server will only serve the models that are "
        "on disk in the configured model path.",
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

    jwt_auth = parser.add_argument_group("JWT Authentication")
    jwt_auth.add_argument(
        "--jwt-secret",
        type=str,
        help="Public key for asymmetric JWT methods or shared secret"
        "for symmetric methods. Please also make sure to use "
        "--jwt-method to select the method of the signature, "
        "otherwise this argument will be ignored.",
    )
    jwt_auth.add_argument(
        "--jwt-method",
        type=str,
        default="HS256",
        help="Method used for the signature of the JWT authentication payload.",
    )
