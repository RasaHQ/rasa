import argparse

from rasa.constants import DEFAULT_DATA_PATH

from rasa.cli.arguments.default_arguments import add_model_param, add_data_param
from rasa.cli.arguments.run import add_server_arguments


def set_x_arguments(parser: argparse.ArgumentParser):
    add_model_param(parser, add_positional_arg=False)

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
        "--nlg",
        type=str,
        default="http://localhost:5002/api/nlg",
        help="Rasa NLG endpoint.",
    )

    parser.add_argument(
        "--model-endpoint-url",
        type=str,
        default="http://localhost:5002/api/projects/default/models/tags/production",
        help="Rasa model endpoint URL.",
    )

    parser.add_argument(
        "--project-path",
        type=str,
        default=".",
        help="Path to the Rasa project directory.",
    )

    add_data_param(parser, default=DEFAULT_DATA_PATH, data_type="stories and Rasa NLU ")

    add_server_arguments(parser)
