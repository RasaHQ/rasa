from __future__ import annotations

import argparse
from pathlib import Path
from typing import Text, Union

import structlog

import rasa.cli.utils
import rasa.shared.utils.cli
from rasa.shared.constants import DEFAULT_CONFIG_PATH, DEFAULT_ENDPOINTS_PATH
from rasa.shared.utils.io import write_text_file
from rasa.studio.data_handler import StudioDataHandler
from rasa.studio.download.download import handle_download
from rasa.studio.link import get_studio_config, read_assistant_name

structlogger = structlog.get_logger(__name__)


def _write_to_file(
    content: Text, file_type: Text, file_path: Text, default_path: Text
) -> None:
    """Write the content to a file.

    Args:
        content: The content to write.
        file_type: The type of file (e.g., "config" or "endpoints".).
        file_path: The path to the file.
        default_path: The default path to use file_path is not valid.
    """
    path: Union[Path, str, None] = rasa.cli.utils.get_validated_path(
        file_path, file_type, default_path, none_is_valid=True
    )
    write_text_file(content, path, encoding="utf-8")
    rasa.shared.utils.cli.print_success(f"Pulled {file_type} data from assistant.")


def handle_pull(args: argparse.Namespace) -> None:
    """Pull all data and overwrite the local assistant.

    Args:
        args: The parsed arguments.
    """
    assistant_name = read_assistant_name()

    # Use the CLI command logic to download with overwrite
    download_args = argparse.Namespace(**vars(args))
    download_args.assistant_name = [assistant_name]
    download_args.overwrite = True

    handle_download(download_args)
    rasa.shared.utils.cli.print_success("Pulled the data from assistant.")


def handle_pull_config(args: argparse.Namespace) -> None:
    """Pull just the assistant's `config.yml`.

    Args:
        args: The parsed arguments.
    """
    studio_cfg = get_studio_config()
    assistant_name = read_assistant_name()

    handler = StudioDataHandler(studio_cfg, assistant_name)
    handler.request_all_data()

    config_yaml = handler.get_config()
    if not config_yaml:
        rasa.shared.utils.cli.print_error_and_exit(
            "No configuration data was found in the Studio assistant."
        )

    _write_to_file(config_yaml, "config", args.config, DEFAULT_CONFIG_PATH)


def handle_pull_endpoints(args: argparse.Namespace) -> None:
    """Pull just the assistant's `endpoints.yml`.

    Args:
        args: The parsed arguments.
    """
    studio_cfg = get_studio_config()
    assistant_name = read_assistant_name()

    handler = StudioDataHandler(studio_cfg, assistant_name)
    handler.request_all_data()

    endpoints_yaml = handler.get_endpoints()
    if not endpoints_yaml:
        rasa.shared.utils.cli.print_error_and_exit(
            "No endpoints data was found in the Studio assistant."
        )

    _write_to_file(endpoints_yaml, "endpoints", args.endpoints, DEFAULT_ENDPOINTS_PATH)
