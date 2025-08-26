import base64
import os
from typing import Optional

import structlog

from rasa.model_manager import config
from rasa.utils.io import subpath

structlogger = structlog.get_logger()


def write_encoded_data_to_file(encoded_data: bytes, file: str) -> None:
    """Write base64 encoded data to a file."""
    # create the directory if it does not exist of the parent directory
    os.makedirs(os.path.dirname(file), exist_ok=True)

    with open(file, "w") as f:
        decoded = base64.b64decode(encoded_data)
        text = decoded.decode("utf-8")
        f.write(text)


def logs_base_path() -> str:
    """Return the path to the logs' directory."""
    return subpath(config.SERVER_BASE_WORKING_DIRECTORY, "logs")


def ensure_base_directory_exists(directory: str) -> None:
    """Ensure that a files parent directory exists.

    Args:
        directory: The directory to check.
    """
    os.makedirs(os.path.dirname(directory), exist_ok=True)


def models_base_path() -> str:
    """Return the path to the models' directory."""
    return subpath(config.SERVER_BASE_WORKING_DIRECTORY, "models")


def logs_path(action_id: str) -> str:
    """Return the path to the log file for a given action id.

    Args:
        action_id: can either be a training_id or a deployment_id
    """
    return subpath(logs_base_path(), f"{action_id}.txt")


def get_logs_content(action_id: str) -> Optional[str]:
    """Return the content of the log file for a given action id."""
    try:
        with open(logs_path(action_id), "r") as file:
            return file.read()
    except FileNotFoundError:
        structlogger.debug("model_service.logs.not_found", action_id=action_id)
        return None
