import argparse
from pathlib import Path

import rasa.shared.utils.cli
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_ENDPOINTS_PATH,
)
from rasa.studio.constants import DOMAIN_FILENAME


def validate_argument_paths(args: argparse.Namespace) -> None:
    """Validates the paths provided in the command line arguments.

    Args:
        args: The command line arguments containing paths to validate.
    """

    def validate_path(arg_name: str, default: str) -> None:
        path_value = getattr(args, arg_name, None)
        if path_value and path_value != default:
            resolved_path = Path(path_value).resolve()
            if not resolved_path.exists():
                rasa.shared.utils.cli.print_error_and_exit(
                    f"{arg_name.capitalize()} file or directory "
                    f"'{path_value}' does not exist."
                )

    validate_path("domain", DOMAIN_FILENAME)
    validate_path("config", DEFAULT_CONFIG_PATH)
    validate_path("endpoints", DEFAULT_ENDPOINTS_PATH)
    validate_path("data", DEFAULT_DATA_PATH)
