import glob
import json
import logging
import os
import shutil
import tarfile
import tempfile
from typing import Text, Tuple, Union, Optional, List, Dict, Any

import rasa.utils.io
from rasa.constants import DEFAULT_MODELS_PATH

# Type alias for the fingerprint
from rasa.core.domain import Domain

Fingerprint = Dict[Text, Union[Text, List[Text]]]

logger = logging.getLogger(__name__)

FINGERPRINT_FILE_PATH = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_DOMAIN_KEY = "domain"
FINGERPRINT_RASA_VERSION_KEY = "version"
FINGERPRINT_STORIES_KEY = "stories"
FINGERPRINT_NLU_DATA_KEY = "messages"
FINGERPRINT_TRAINED_AT_KEY = "trained_at"


def get_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Gets a model and unpacks it.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Returns:
        Path to the unpacked model.

    """
    if not model_path:
        return None
    elif os.path.isdir(model_path):
        model_path = get_latest_model(model_path)

    if model_path:
        return unpack_model(model_path)

    return None


def get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Gets the latest model from a path.

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


def add_evaluation_file_to_model(
    model_path: Text, payload: Union[Text, Dict[Text, Any]], data_format: Text = "json"
) -> Text:
    """Adds NLU data `payload` to zipped model at `model_path`.

    Args:
        model_path: Path to zipped Rasa Stack model.
        payload: Json payload to be added to the Rasa Stack model.
        data_format: NLU data format of `payload` ('json' or 'md').

    Returns:
        Path of the new archive in a temporary directory.
    """

    # create temporary directory
    tmpdir = tempfile.mkdtemp()

    # unpack archive
    _ = unpack_model(model_path, tmpdir)

    # add model file to folder
    if data_format == "json":
        data_path = os.path.join(tmpdir, "data.json")
        with open(data_path, "w") as f:
            f.write(json.dumps(payload))
    elif data_format == "md":
        data_path = os.path.join(tmpdir, "nlu.md")
        with open(data_path, "w") as f:
            f.write(payload)
    else:
        raise ValueError("`data_format` needs to be either `md` or `json`.")

    zipped_path = os.path.join(tmpdir, os.path.basename(model_path))

    # re-archive and post
    with tarfile.open(zipped_path, "w:gz") as tar:
        for elem in os.scandir(tmpdir):
            tar.add(elem.path, arcname=elem.name)

    return zipped_path


def unpack_model(model_file: Text, working_directory: Optional[Text] = None) -> Text:
    """Unpacks a zipped Rasa model.

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

    tar = tarfile.open(model_file)

    # cast `working_directory` as str for py3.5 compatibility
    working_directory = str(working_directory)

    # All files are in a subdirectory.
    tar.extractall(working_directory)
    tar.close()
    logger.debug("Extracted model to '{}'.".format(working_directory))

    return working_directory


def get_model_subdirectories(unpacked_model_path: Text) -> Tuple[Text, Text]:
    """Returns paths for core and nlu model directories.

    Args:
        unpacked_model_path: Path to unpacked Rasa model.

    Returns:
        Tuple (path to Core subdirectory, path to NLU subdirectory).
    """
    core_path = os.path.join(unpacked_model_path, "core")
    nlu_path = os.path.join(unpacked_model_path, "nlu")

    return core_path, nlu_path


def create_package_rasa(
    training_directory: Text,
    output_filename: Text,
    fingerprint: Optional[Fingerprint] = None,
) -> Text:
    """Creates a zipped Rasa model from trained model files.

    Args:
        training_directory: Path to the directory which contains the trained
                            model files.
        output_filename: Name of the zipped model file to be created.
        fingerprint: A unique fingerprint to identify the model version.

    Returns:
        Path to zipped model.

    """
    import tarfile

    if fingerprint:
        persist_fingerprint(training_directory, fingerprint)

    output_directory = os.path.dirname(output_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with tarfile.open(output_filename, "w:gz") as tar:
        for elem in os.scandir(training_directory):
            tar.add(elem.path, arcname=elem.name)

    shutil.rmtree(training_directory)
    return output_filename


def model_fingerprint(
    config_file: Text,
    domain: Optional[Union[Domain, Text]] = None,
    nlu_data: Optional[Text] = None,
    stories: Optional[Text] = None,
) -> Fingerprint:
    """Creates a model fingerprint from its used configuration and training
    data.

    Args:
        config_file: Path to the configuration file.
        domain: Path to the models domain file.
        nlu_data: Path to the used NLU training data.
        stories: Path to the used story training data.

    Returns:
        The fingerprint.

    """
    import rasa
    import time

    if isinstance(domain, Domain):
        domain_hash = hash(domain)
    else:
        domain_hash = _get_hashes_for_paths(domain)

    return {
        FINGERPRINT_CONFIG_KEY: _get_hashes_for_paths(config_file),
        FINGERPRINT_DOMAIN_KEY: domain_hash,
        FINGERPRINT_NLU_DATA_KEY: _get_hashes_for_paths(nlu_data),
        FINGERPRINT_STORIES_KEY: _get_hashes_for_paths(stories),
        FINGERPRINT_TRAINED_AT_KEY: time.time(),
        FINGERPRINT_RASA_VERSION_KEY: rasa.__version__,
    }


def _get_hashes_for_paths(path: Text) -> List[Text]:
    from rasa.core.utils import get_file_hash

    files = []
    if path and os.path.isdir(path):
        files = [
            os.path.join(path, f) for f in os.listdir(path) if not f.startswith(".")
        ]
    elif path and os.path.isfile(path):
        files = [path]

    return sorted([get_file_hash(f) for f in files])


def fingerprint_from_path(model_path: Text) -> Fingerprint:
    """Loads a persisted fingerprint.

    Args:
        model_path: Path to directory containing the fingerprint.

    Returns:
        The fingerprint or an empty dict if no fingerprint was found.
    """
    if not model_path or not os.path.exists(model_path):
        return {}

    fingerprint_path = os.path.join(model_path, FINGERPRINT_FILE_PATH)

    if os.path.isfile(fingerprint_path):
        return rasa.utils.io.read_json_file(fingerprint_path)
    else:
        return {}


def persist_fingerprint(output_path: Text, fingerprint: Fingerprint):
    """Persists a model fingerprint.

    Args:
        output_path: Directory in which the fingerprint should be saved.
        fingerprint: The fingerprint to be persisted.

    """
    from rasa.core.utils import dump_obj_as_json_to_file

    path = os.path.join(output_path, FINGERPRINT_FILE_PATH)
    dump_obj_as_json_to_file(path, fingerprint)


def core_fingerprint_changed(
    fingerprint1: Fingerprint, fingerprint2: Fingerprint
) -> bool:
    """Checks whether the fingerprints of the Core model changed.

    Args:
        fingerprint1: A fingerprint.
        fingerprint2: Another fingerprint.

    Returns:
        `True` if the fingerprint for the Core model changed, else `False`.

    """
    relevant_keys = [
        FINGERPRINT_CONFIG_KEY,
        FINGERPRINT_DOMAIN_KEY,
        FINGERPRINT_STORIES_KEY,
        FINGERPRINT_RASA_VERSION_KEY,
    ]

    for k in relevant_keys:
        if fingerprint1.get(k) != fingerprint2.get(k):
            logger.info("Data ({}) for dialogue model changed.".format(k))
            return True
    return False


def nlu_fingerprint_changed(
    fingerprint1: Fingerprint, fingerprint2: Fingerprint
) -> bool:
    """Checks whether the fingerprints of the NLU model changed.

    Args:
        fingerprint1: A fingerprint.
        fingerprint2: Another fingerprint.

    Returns:
        `True` if the fingerprint for the NLU model changed, else `False`.

    """

    relevant_keys = [
        FINGERPRINT_CONFIG_KEY,
        FINGERPRINT_NLU_DATA_KEY,
        FINGERPRINT_RASA_VERSION_KEY,
    ]

    for k in relevant_keys:
        if fingerprint1.get(k) != fingerprint2.get(k):
            logger.info("Data ({}) for NLU model changed.".format(k))
            return True
    return False


def merge_model(source: Text, target: Text) -> bool:
    """Merges two model directories.

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
        logging.debug(e)
        return False
