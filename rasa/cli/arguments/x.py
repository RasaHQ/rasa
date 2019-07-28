import argparse

from rasa.constants import DEFAULT_DATA_PATH, DEFAULT_RASA_X_PORT

from rasa.cli.arguments.default_arguments import add_model_param, add_data_param
from rasa.cli.arguments.run import add_server_arguments


def set_x_arguments(parser: argparse.ArgumentParser):
    add_model_param(parser, add_positional_arg=False)

    add_data_param(parser, default=DEFAULT_DATA_PATH, data_type="stories and Rasa NLU ")

    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatic yes or default options to prompts and oppressed warnings.",
    )

    parser.add_argument(
        "--production",
        action="store_true",
        help="Run Rasa X in a production environment.",
    )

    parser.add_argument(
        "--rasa-x-port",
        default=DEFAULT_RASA_X_PORT,
        type=int,
        help="Port to run the Rasa X server at.",
    )

    add_server_arguments(parser)
