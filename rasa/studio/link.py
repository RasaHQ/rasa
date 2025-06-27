from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

import questionary
import structlog
from pydantic import BaseModel, Field

import rasa.shared.utils.cli
from rasa.constants import RASA_DIR_NAME
from rasa.shared.utils.yaml import read_yaml_file, write_yaml
from rasa.studio.config import StudioConfig
from rasa.studio.upload import (
    check_if_assistant_already_exists,
    handle_upload,
    is_auth_working,
)

structlogger = structlog.get_logger(__name__)

_LINK_FILE_NAME: Text = "studio.yml"


class AssistantLinkPayload(BaseModel):
    assistant_name: Text
    studio_url: Text
    linked_at: Text = Field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )


def _link_file(project_root: Path) -> Path:
    """Return `<project-root>/.rasa/studio.yml`.

    Args:
        project_root: The path to the project root.

    Returns:
        The path to the link file.
    """
    return project_root / RASA_DIR_NAME / _LINK_FILE_NAME


def _write_link_file(
    project_root: Path, assistant_name: Text, studio_url: Text
) -> None:
    """Persist assistant information inside the project.

    Args:
        project_root: The path to the project root.
        assistant_name: The name of the assistant.
        studio_url: The URL of the Rasa Studio instance.
    """
    file_path = _link_file(project_root)
    file_path.parent.mkdir(exist_ok=True, parents=True)

    payload = AssistantLinkPayload(
        assistant_name=assistant_name,
        studio_url=studio_url,
    )
    write_yaml(payload.model_dump(), file_path)


def _read_link_file(
    project_root: Path = Path.cwd(),
) -> Optional[Union[List[Any], Dict[Text, Any]]]:
    """Reads the link configuration file.

    Args:
        project_root: The path to the project root.

    Returns:
        The assistant information if the file exists, otherwise None.
    """
    file_path = _link_file(project_root)
    if not file_path.is_file():
        return None

    return read_yaml_file(file_path)


def read_assistant_name(project_root: Path = Path.cwd()) -> Optional[Text]:
    """Reads the assistant_name from the linked configuration file.

    Args:
        project_root: The path to the project root.

    Returns:
        The assistant name if the file exists, otherwise None.
    """
    linked = _read_link_file(project_root)
    assistant_name = (
        linked.get("assistant_name") if linked and isinstance(linked, dict) else None
    )

    if not assistant_name:
        rasa.shared.utils.cli.print_error_and_exit(
            "This project is not linked to any Rasa Studio assistant.\n"
            "Run `rasa studio link <assistant-name>` first."
        )

    return assistant_name


def get_studio_config() -> StudioConfig:
    """Get the StudioConfig object or exit with an error message.

    Returns:
        A valid StudioConfig object.
    """
    config = StudioConfig.read_config()
    if not config.is_valid():
        rasa.shared.utils.cli.print_error_and_exit(
            "Rasa Studio is not configured correctly. Run `rasa studio config` first."
        )
    if not is_auth_working(config.studio_url, not config.disable_verify):
        rasa.shared.utils.cli.print_error_and_exit(
            "Authentication invalid or expired. Please run `rasa studio login`."
        )
    return config


def _ensure_assistant_exists(
    assistant_name: Text,
    studio_cfg: StudioConfig,
    args: argparse.Namespace,
) -> bool:
    """Create the assistant on Studio if it does not yet exist.

    Args:
        assistant_name: The name the user provided on the CLI.
        studio_cfg: The validated Studio configuration.
        args: The original CLI args (needed for `handle_upload`).

    Returns:
        True if the assistant already exists or was created, False otherwise.
    """
    verify_ssl = not studio_cfg.disable_verify
    assistant_already_exists = check_if_assistant_already_exists(
        assistant_name, studio_cfg.studio_url, verify_ssl
    )
    if not assistant_already_exists:
        should_create_assistant = questionary.confirm(
            f"Assistant '{assistant_name}' was not found on Rasa Studio. "
            f"Do you want to create it?"
        ).ask()
        if should_create_assistant:
            # `handle_upload` expects the name to live in `args.assistant_name`
            args.assistant_name = assistant_name
            handle_upload(args)

            rasa.shared.utils.cli.print_info(
                f"Assistant {assistant_name} successfully created."
            )
        return should_create_assistant

    return assistant_already_exists


def handle_link(args: argparse.Namespace) -> None:
    """Implementation of `rasa studio link <assistant-name>` CLI command.

    Args:
        args: The command line arguments.
    """
    assistant_name: Text = args.assistant_name
    studio_cfg = get_studio_config()
    assistant_exists = _ensure_assistant_exists(assistant_name, studio_cfg, args)
    if not assistant_exists:
        rasa.shared.utils.cli.print_error_and_exit(
            "Project has not been linked with Studio assistant."
        )

    project_root = Path.cwd()
    link_file = _link_file(project_root)

    if link_file.exists():
        linked_assistant_name = read_assistant_name(project_root)
        if linked_assistant_name == assistant_name:
            rasa.shared.utils.cli.print_info(
                f"Project is already linked to assistant '{assistant_name}'."
            )
            sys.exit(0)

        overwrite = questionary.confirm(
            f"Project is currently linked to the following Rasa Studio assistant:\n\n"
            f"  Assistant name: {linked_assistant_name}\n"
            f"  Studio URL: {studio_cfg.studio_url}\n"
            f"  Keycloak Auth URL: {studio_cfg.authentication_server_url}\n\n"
            f"Do you want to overwrite it with the new assistant '{assistant_name}'?"
        ).ask()
        if not overwrite:
            rasa.shared.utils.cli.print_info(
                "Existing link kept – nothing was changed."
            )
            sys.exit(0)

    _write_link_file(project_root, assistant_name, studio_cfg.studio_url)

    structlogger.info(
        "studio.link.success",
        event_info=f"Project linked to Studio assistant '{assistant_name}'.",
        assistant_name=assistant_name,
    )
    rasa.shared.utils.cli.print_success(
        f"Project successfully linked to assistant '{assistant_name}'."
    )
