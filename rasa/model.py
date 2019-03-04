import logging
import os
import glob
import shutil
import tempfile
from typing import Text, Tuple, Union, Optional, List, Dict

from rasa.constants import DEFAULT_MODELS_PATH

# Type alias for the fingerprint
Fingerprint = Dict[Text, Union[Text, List[Text]]]

logger = logging.getLogger(__name__)

FINGERPRINT_FILE_PATH = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_DOMAIN_KEY = "domain"
FINGERPRINT_NLU_VERSION_KEY = "nlu_version"
FINGERPRINT_CORE_VERSION_KEY = "core_version"
FINGERPRINT_RASA_VERSION_KEY = "version"
FINGERPRINT_STORIES_KEY = "stories"
FINGERPRINT_NLU_DATA_KEY = "messages"


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

    return unpack_model(model_path)


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


def unpack_model(model_file: Text, working_directory: Text = None,
                 ) -> Union[Text, Tuple[Text, Text, Text]]:
    """Unpacks a zipped Rasa model.

    Args:
        model_file: Path to zipped model.
        working_directory: Location where the model should be unpacked to.
                           If `None` a tempory directory will be created.

    Returns:
        Path to unpacked Rasa model.

    """
    import tarfile

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    tar = tarfile.open(model_file)
    tar.extractall(working_directory)
    tar.close()
    logger.debug("Extracted model to '{}'.".format(working_directory))

    return os.path.join(working_directory, "rasa_model")


def get_model_subdirectories(unpacked_model_path: Text) -> Tuple[Text, Text]:
    """Returns paths for core and nlu model directories.

    Args:
        unpacked_model_path: Path to unpacked Rasa model.

    Returns:
        Tuple (path to Core subdirectory, path to NLU subdirectory).
    """
    return (os.path.join(unpacked_model_path, "core"),
            os.path.join(unpacked_model_path, "nlu"))


def create_package_rasa(training_directory: Text, model_directory: Text,
                        output_filename: Text,
                        fingerprint: Optional[Fingerprint] = None) -> Text:
    """Creates a zipped Rasa model from trained model files.

    Args:
        training_directory: Path to the directory which contains the trained
                            model files.
        model_directory: Name of the subdirectory in the zipped file which
                         should contain the trained models.
        output_filename: Name of the zipped model file to be created.
        fingerprint: A unique fingerprint to identify the model version.

    Returns:
        Path to zipped model.

    """
    import tarfile

    full_path = os.path.join(training_directory, model_directory)

    if fingerprint:
        persist_fingerprint(full_path, fingerprint)

    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))

    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(full_path, arcname=os.path.basename(full_path))

    shutil.rmtree(training_directory)
    return output_filename


def model_fingerprint(config_file: Text, domain_file: Optional[Text] = None,
                      nlu_data: Optional[Text] = None,
                      stories: Optional[Text] = None
                      ) -> Fingerprint:
    """Creates a model fingerprint from its used configuration and training
    data.

    Args:
        config_file: Path to the configuration file.
        domain_file: Path to the models domain file.
        nlu_data: Path to the used NLU training data.
        stories: Path to the used story training data.

    Returns:
        The fingerprint.

    """
    import rasa_core
    import rasa_nlu
    import rasa

    return {
        FINGERPRINT_CONFIG_KEY: _get_hashes_for_paths(config_file),
        FINGERPRINT_DOMAIN_KEY: _get_hashes_for_paths(domain_file),
        FINGERPRINT_NLU_DATA_KEY: _get_hashes_for_paths(nlu_data),
        FINGERPRINT_STORIES_KEY: _get_hashes_for_paths(stories),
        FINGERPRINT_NLU_VERSION_KEY: rasa_nlu.__version__,
        FINGERPRINT_CORE_VERSION_KEY: rasa_core.__version__,
        FINGERPRINT_RASA_VERSION_KEY: rasa.__version__
    }


def _get_hashes_for_paths(path: Text) -> List[Text]:
    from rasa_core.utils import get_file_hash

    files = []
    if path and os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if not f.startswith('.')]
    elif path and os.path.isfile(path):
        files = [path]

    return [get_file_hash(f) for f in files]


def fingerprint_from_path(model_path: Text) -> Fingerprint:
    """Loads a persisted fingerprint.

    Args:
        model_path: Path to directory containing the fingerprint.

    Returns:
        The fingerprint or an empty dict if no fingerprint was found.
    """
    import rasa_core

    fingerprint_path = os.path.join(model_path, FINGERPRINT_FILE_PATH)

    if os.path.isfile(fingerprint_path):
        return rasa_core.utils.read_json_file(fingerprint_path)
    else:
        return {}


def persist_fingerprint(output_path: Text, fingerprint: Fingerprint):
    """Persists a model fingerprint.

    Args:
        output_path: Directory in which the fingerprint should be saved.
        fingerprint: The fingerprint to be persisted.

    """
    from rasa_core.utils import dump_obj_as_json_to_file

    path = os.path.join(output_path, FINGERPRINT_FILE_PATH)
    dump_obj_as_json_to_file(path, fingerprint)


def core_fingerprint_changed(fingerprint1: Fingerprint,
                             fingerprint2: Fingerprint) -> bool:
    """Checks whether the fingerprints of the Core model changed.

    Args:
        fingerprint1: A fingerprint.
        fingerprint2: Another fingerprint.

    Returns:
        `True` if the fingerprint for the Core model changed, else `False`.

    """
    relevant_keys = [FINGERPRINT_CONFIG_KEY, FINGERPRINT_CORE_VERSION_KEY,
                     FINGERPRINT_DOMAIN_KEY, FINGERPRINT_STORIES_KEY,
                     FINGERPRINT_RASA_VERSION_KEY]

    return any(
        [fingerprint1.get(k) != fingerprint2.get(k) for k in relevant_keys])


def nlu_fingerprint_changed(fingerprint1: Fingerprint,
                            fingerprint2: Fingerprint) -> bool:
    """Checks whether the fingerprints of the NLU model changed.

    Args:
        fingerprint1: A fingerprint.
        fingerprint2: Another fingerprint.

    Returns:
        `True` if the fingerprint for the NLU model changed, else `False`.

    """
    relevant_keys = [FINGERPRINT_CONFIG_KEY, FINGERPRINT_NLU_VERSION_KEY,
                     FINGERPRINT_NLU_DATA_KEY, FINGERPRINT_RASA_VERSION_KEY]

    return any(
        [fingerprint1.get(k) != fingerprint2.get(k) for k in relevant_keys])


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
