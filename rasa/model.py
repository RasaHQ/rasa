import glob
import logging
import os
import shutil
import tempfile
from typing import Text, Tuple, Union, Optional, List, Dict

import yaml.parser

import rasa.utils.io
from rasa.constants import (
    DEFAULT_MODELS_PATH,
    CONFIG_MANDATORY_KEYS_CORE,
    CONFIG_MANDATORY_KEYS_NLU,
    CONFIG_MANDATORY_KEYS,
)

# Type alias for the fingerprint
from rasa.core import config
from rasa.core.domain import Domain
from rasa.core.utils import get_dict_hash

Fingerprint = Dict[Text, Union[Text, List[Text]]]

logger = logging.getLogger(__name__)

FINGERPRINT_FILE_PATH = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_CONFIG_CORE_KEY = "core-config"
FINGERPRINT_CONFIG_NLU_KEY = "nlu-config"
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
        FINGERPRINT_CONFIG_KEY: _get_hash_of_config(
            config_file, exclude_keys=CONFIG_MANDATORY_KEYS
        ),
        FINGERPRINT_CONFIG_CORE_KEY: _get_hash_of_config(
            config_file, include_keys=CONFIG_MANDATORY_KEYS_CORE
        ),
        FINGERPRINT_CONFIG_NLU_KEY: _get_hash_of_config(
            config_file, include_keys=CONFIG_MANDATORY_KEYS_NLU
        ),
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


def _get_hash_of_config(
    config_path: Text,
    include_keys: Optional[List[Text]] = None,
    exclude_keys: Optional[List[Text]] = None,
) -> Text:
    if not config_path or not os.path.exists(config_path):
        return ""

    try:
        config_dict = rasa.utils.io.read_yaml_file(config_path)
    except yaml.parser.ParserError as e:
        logger.debug(
            "Failed to read config file '{}'. Error: {}".format(config_path, e)
        )
        return ""

    keys = include_keys or list(
        filter(lambda k: k not in exclude_keys, config_dict.keys())
    )

    sub_config = dict((k, config_dict[k]) for k in keys if k in config_dict)

    return get_dict_hash(sub_config)


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
        FINGERPRINT_CONFIG_CORE_KEY,
        FINGERPRINT_DOMAIN_KEY,
        FINGERPRINT_STORIES_KEY,
        FINGERPRINT_RASA_VERSION_KEY,
    ]

    for k in relevant_keys:
        if fingerprint1.get(k) != fingerprint2.get(k):
            logger.info("Data ({}) for Core model changed.".format(k))
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
        FINGERPRINT_CONFIG_NLU_KEY,
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


def should_retrain(new_fingerprint: Fingerprint, old_model: Text, train_path: Text):
    """Checks which component of a model should be retrained.

    Args:
        new_fingerprint: The fingerprint of the new model to be trained.
        old_model: Path to the old zipped model file.
        train_path: Path to the directory in which the new model will be trained.

    Returns:
        A tuple of boolean values indicating whether Rasa Core and/or Rasa NLU needs
        to be retrained or not.

    """
    retrain_nlu = retrain_core = True

    if old_model is None or not os.path.exists(old_model):
        return retrain_core, retrain_nlu

    unpacked = unpack_model(old_model)
    last_fingerprint = fingerprint_from_path(unpacked)

    old_core, old_nlu = get_model_subdirectories(unpacked)

    if not core_fingerprint_changed(last_fingerprint, new_fingerprint):
        target_path = os.path.join(train_path, "core")
        retrain_core = not merge_model(old_core, target_path)

    if not nlu_fingerprint_changed(last_fingerprint, new_fingerprint):
        target_path = os.path.join(train_path, "nlu")
        retrain_nlu = not merge_model(old_nlu, target_path)

    return retrain_core, retrain_nlu
