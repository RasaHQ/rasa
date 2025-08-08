import argparse
from pathlib import Path
from typing import List

import rasa.shared.utils.cli

DOMAIN_FILENAME = "domain.yml"
DEFAULT_CONFIG_PATH = "config.yml"
DEFAULT_ENDPOINTS_PATH = "endpoints.yml"
DEFAULT_DATA_PATH = "data"


def validate_argument_paths(args: argparse.Namespace) -> None:
    """Validate every path passed via CLI arguments.

    Args:
        args: CLI arguments containing paths to validate.

    Raises:
        rasa.shared.utils.cli.PrintErrorAndExit: If any path does not exist.
    """
    invalid_paths: List[str] = []

    def collect_invalid_paths(arg_name: str, default: str) -> None:
        value = getattr(args, arg_name, None)
        path_values = value if isinstance(value, list) else [value]
        for path_value in path_values:
            if not path_value or path_value == default:
                continue

            if not Path(path_value).resolve().exists():
                invalid_paths.append(f"{arg_name}: '{path_value}'")

    collect_invalid_paths("domain", DOMAIN_FILENAME)
    collect_invalid_paths("config", DEFAULT_CONFIG_PATH)
    collect_invalid_paths("endpoints", DEFAULT_ENDPOINTS_PATH)
    collect_invalid_paths("data", DEFAULT_DATA_PATH)

    if invalid_paths:
        message = (
            "The following files or directories do not exist:\n  - "
            + "\n  - ".join(invalid_paths)
        )
        rasa.shared.utils.cli.print_error_and_exit(message)
