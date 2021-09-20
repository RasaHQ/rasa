import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import (
    Text,
    Tuple,
    Union,
    Optional,
)

from rasa.shared.constants import (
    DEFAULT_MODELS_PATH,
    DEFAULT_CORE_SUBDIRECTORY_NAME,
    DEFAULT_NLU_SUBDIRECTORY_NAME,
)

from rasa.exceptions import ModelNotFound
from rasa.utils.common import TempDirectoryPath


logger = logging.getLogger(__name__)


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
    elif not os.path.exists(model_path):
        raise ModelNotFound(f"No file or directory at '{model_path}'.")

    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)
        if not model_path:
            raise ModelNotFound(
                f"Could not find any Rasa model files in '{model_path}'."
            )
    elif not model_path.endswith(".tar.gz"):
        raise ModelNotFound(f"Path '{model_path}' does not point to a Rasa model file.")

    return model_path


def get_model(model_path: Text = DEFAULT_MODELS_PATH) -> TempDirectoryPath:
    """Gets a model and unpacks it.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Returns:
        Path to the unpacked model.

    Raises:
        ModelNotFound Exception: When no model could be found at the provided path.

    """
    model_path = get_local_model(model_path)

    try:
        model_relative_path = os.path.relpath(model_path)
    except ValueError:
        model_relative_path = model_path

    logger.info(f"Loading model {model_relative_path}...")

    return unpack_model(model_path)


def get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Get the latest model from a path.

    Args:
        model_path: Path to a directory containing zipped models.

    Returns:
        Path to latest model in the given directory.

    """
    if not os.path.exists(model_path) or os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)

    list_of_files = glob.glob(os.path.join(model_path, "*.tar.gz"))

    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)


def unpack_model(
    model_file: Text, working_directory: Optional[Union[Path, Text]] = None
) -> TempDirectoryPath:
    """Unpack a zipped Rasa model.

    Args:
        model_file: Path to zipped model.
        working_directory: Location where the model should be unpacked to.
                           If `None` a temporary directory will be created.

    Returns:
        Path to unpacked Rasa model.

    """
    import tarfile

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    # All files are in a subdirectory.
    try:
        with tarfile.open(model_file, mode="r:gz") as tar:
            tar.extractall(working_directory)
            logger.debug(f"Extracted model to '{working_directory}'.")
    except (tarfile.TarError, ValueError) as e:
        logger.error(f"Failed to extract model at {model_file}. Error: {e}")
        raise

    return TempDirectoryPath(working_directory)


def get_model_subdirectories(
    unpacked_model_path: Text,
) -> Tuple[Optional[Text], Optional[Text]]:
    """Return paths for Core and NLU model directories, if they exist.
    If neither directories exist, a `ModelNotFound` exception is raised.

    Args:
        unpacked_model_path: Path to unpacked Rasa model.

    Returns:
        Tuple (path to Core subdirectory if it exists or `None` otherwise,
               path to NLU subdirectory if it exists or `None` otherwise).

    """
    core_path = os.path.join(unpacked_model_path, DEFAULT_CORE_SUBDIRECTORY_NAME)
    nlu_path = os.path.join(unpacked_model_path, DEFAULT_NLU_SUBDIRECTORY_NAME)

    if not os.path.isdir(core_path):
        core_path = None

    if not os.path.isdir(nlu_path):
        nlu_path = None

    if not core_path and not nlu_path:
        raise ModelNotFound(
            "No NLU or Core data for unpacked model at: '{}'.".format(
                unpacked_model_path
            )
        )

    return core_path, nlu_path


def move_model(source: Text, target: Text) -> bool:
    """Move two model directories.

    Args:
        source: The original folder which should be merged in another.
        target: The destination folder where it should be moved to.

    Returns:
        `True` if the merge was successful, else `False`.

    """
    try:
        shutil.move(source, target)
        return True
    except Exception as e:
        logging.debug(f"Could not merge model: {e}")
        return False


def get_model_for_finetuning(
    previous_model_file: Optional[Union[Path, Text]]
) -> Optional[Union[Path, Text]]:
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
        return get_latest_model(previous_model_file)

    if Path(previous_model_file).is_file():
        return previous_model_file

    logger.debug(
        "No valid model for finetuning found as directory either "
        "contains no model or model file cannot be found."
    )
    return None
