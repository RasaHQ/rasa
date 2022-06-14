import argparse
from rasa.cli.arguments import default_arguments
from rasa.cli.arguments.run import add_server_arguments


def set_x_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for running rasa x --production."""
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run a Rasa server in a mode that allows connecting "
        "to Rasa Enterprise as the config endpoint",
    )

    parser.add_argument(
        "--config-endpoint",
        type=str,
        help="Rasa X endpoint URL from which to pull the runtime config. This URL "
        "typically contains the Rasa X token for authentication. Example: "
        "https://example.com/api/config?token=my_rasa_x_token",
    )

    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatic yes or default options to prompts and oppressed warnings.",
    )

    default_arguments.add_model_param(parser, add_positional_arg=False)
    add_server_arguments(parser)
