import glob
import hashlib
import logging
import os
from pathlib import Path
from subprocess import check_output, DEVNULL, CalledProcessError
from typing import (
    Text,
    Optional,
    Union,
)

from rasa.shared.constants import DEFAULT_MODELS_PATH

from rasa.exceptions import ModelNotFound


logger = logging.getLogger(__name__)

# TODO: rename this whole module.


def get_local_model(model_path: Text = DEFAULT_MODELS_PATH) -> Text:
    """Returns verified path to local model archive.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Returns:
        Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Raises:
        ModelNotFound Exception: When no model could be found at the provided path.

    """
    if not model_path:
        raise ModelNotFound("No path specified.")
    elif not Path(model_path).exists():
        raise ModelNotFound(f"No file or directory at '{model_path}'.")

    if Path(model_path).is_dir():
        model_path = get_latest_model(model_path)
        if not model_path:
            raise ModelNotFound(
                f"Could not find any Rasa model files in '{model_path}'."
            )
    elif not model_path.endswith(".tar.gz"):
        raise ModelNotFound(f"Path '{model_path}' does not point to a Rasa model file.")

    return model_path


def get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Get the latest model from a path.

    Args:
        model_path: Path to a directory containing zipped models.

    Returns:
        Path to latest model in the given directory.

    """
    if not model_path:
        return None

    if not Path(model_path).exists() or Path(model_path).is_file():
        model_path = Path(model_path).parent

    list_of_files = glob.glob(Path(model_path).joinpath("*.tar.gz"))

    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)


def get_model_for_finetuning(previous_model_file: Union[Path, Text]) -> Optional[Path]:
    """Gets validated path for model to finetune.

    Args:
        previous_model_file: Path to model file which should be used for finetuning or
            a directory in case the latest trained model should be used.

    Returns:
        Path to model archive. `None` if there is no model.
    """
    if Path(previous_model_file).is_dir():
        logger.debug(
            f"Trying to load latest model from '{previous_model_file}' for "
            f"finetuning."
        )
        previous_model_file = get_latest_model(previous_model_file)

    if previous_model_file and Path(previous_model_file).is_file():
        return Path(previous_model_file)

    logger.debug(
        "No valid model for finetuning found as directory either "
        "contains no model or model file cannot be found."
    )
    return None


def project_fingerprint() -> Optional[Text]:
    """Create a hash for the project in the current working directory.

    Returns:
        project hash
    """
    try:
        remote = check_output(  # skipcq:BAN-B607,BAN-B603
            ["git", "remote", "get-url", "origin"], stderr=DEVNULL
        )
        return hashlib.sha256(remote).hexdigest()
    except (CalledProcessError, OSError):
        return None
