import argparse

from rasa.cli.arguments.default_arguments import add_model_param
from rasa.cli.arguments.run import add_server_arguments


def set_x_arguments(shell_parser: argparse.ArgumentParser):
    add_server_arguments(shell_parser)
    add_model_param(shell_parser, add_positional_arg=False)

    shell_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatic yes or default options to prompts and oppressed warnings.",
    )

    shell_parser.add_argument(
        "--production",
        action="store_true",
        help="Run Rasa X in a production environment.",
    )

    shell_parser.add_argument("--auth-token", type=str, help="Rasa API auth token")

    shell_parser.add_argument(
        "--nlg",
        type=str,
        default="http://localhost:5002/api/nlg",
        help="Rasa NLG endpoint.",
    )

    shell_parser.add_argument(
        "--model-endpoint-url",
        type=str,
        default="http://localhost:5002/api/projects/default/models/tags/production",
        help="Rasa model endpoint URL.",
    )

    shell_parser.add_argument(
        "--project-path",
        type=str,
        default=".",
        help="Path to the Rasa project directory.",
    )

    shell_parser.add_argument(
        "--data-path",
        type=str,
        default="data",
        help=(
            "Path to the directory containing Rasa NLU training data "
            "and Rasa Core stories."
        ),
    )
