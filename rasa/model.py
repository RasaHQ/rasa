import logging
import os
import glob
import shutil
import tempfile
from typing import Text, Tuple, Union, Optional, List, Dict

from rasa_core.utils import get_file_hash, dump_obj_as_json_to_file
import rasa_core
import rasa_nlu
import rasa

Fingerprint = Dict[Text, Union[Text, List[Text]]]

logger = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = "models/"
DEFAULT_DATA_PATH = "data"
DEFAULTS_NLU_DATA_PATH = os.path.join(DEFAULT_DATA_PATH, "nlu")

FINGERPRINT_FILE_PATH = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_DOMAIN_KEY = "domain"
FINGERPRINT_NLU_VERSION_KEY = "nlu_version"
FINGERPRINT_CORE_VERSION_KEY = "core_version"
FINGERPRINT_RASA_VERSION_KEY = "version"
FINGERPRINT_STORIES_KEY = "stories"
FINGERPRINT_NLU_DATA_KEY = "messages"


def get_model(model_path: Text, subdirectories: bool = False) -> Text:
    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)

    return unpack_model(model_path, subdirectories=subdirectories)


def get_latest_model(model_path: Text) -> Optional[Text]:
    if not os.path.exists(model_path) or os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)

    list_of_files = glob.glob(os.path.join(model_path, "*.tar.gz"))

    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)


def unpack_model(model_file: Text, working_directory: Text = None,
                 subdirectories: bool = False
                 ) -> Union[Text, Tuple[Text, Text, Text]]:
    import tarfile

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    tar = tarfile.open(model_file)
    tar.extractall(working_directory)
    tar.close()
    logger.debug("Extracted model to '{}'.".format(working_directory))

    model_directory = os.path.join(working_directory, "rasa_model")

    if not subdirectories:
        return model_directory
    else:
        nlu_subdirectory = os.path.join(model_directory, "nlu")
        core_subdirectory = os.path.join(model_directory, "core")
        return model_directory, core_subdirectory, nlu_subdirectory


def create_package_rasa(training_directory: Text, model_directory: Text,
                        output_filename: Text,
                        fingerprint: Optional[Fingerprint] = None) -> Text:
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
    files = []
    if path and os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path)
                 if not f.startswith('.')]
    elif path and os.path.isfile(path):
        files = [path]

    return [get_file_hash(f) for f in files]


def fingerprint_from_path(model_path: Text) -> Fingerprint:
    fingerprint_path = os.path.join(model_path, FINGERPRINT_FILE_PATH)

    if os.path.isfile(fingerprint_path):
        return rasa_core.utils.read_json_file(fingerprint_path)
    else:
        return {}


def persist_fingerprint(output_path: Text, fingerprint: Fingerprint):
    path = os.path.join(output_path, FINGERPRINT_FILE_PATH)
    dump_obj_as_json_to_file(path, fingerprint)


def core_fingerprint_changed(fingerprint1: Fingerprint,
                             fingerprint2: Fingerprint) -> bool:
    relevant_keys = [FINGERPRINT_CONFIG_KEY, FINGERPRINT_CORE_VERSION_KEY,
                     FINGERPRINT_DOMAIN_KEY, FINGERPRINT_STORIES_KEY]

    return any(
        [fingerprint1.get(k) != fingerprint2.get(k) for k in relevant_keys])


def nlu_fingerprint_changed(fingerprint1: Fingerprint,
                            fingerprint2: Fingerprint) -> bool:
    relevant_keys = [FINGERPRINT_CONFIG_KEY, FINGERPRINT_NLU_VERSION_KEY,
                     FINGERPRINT_NLU_DATA_KEY]

    return any(
        [fingerprint1.get(k) != fingerprint2.get(k) for k in relevant_keys])


def merge_model(source: Text, target: Text) -> bool:
    try:
        shutil.move(source, target)
        return True
    except Exception as e:
        logging.debug(e)
        return False

