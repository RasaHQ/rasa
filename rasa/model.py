import logging
import os
import glob
import shutil
import tempfile
from typing import Text

from rasa_core.utils import get_file_hash, dump_obj_as_json_to_file
import rasa_core
import rasa_nlu
import rasa

logger = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = "models/"
DEFAULT_DATA_PATH = "data"
DEFAULTS_NLU_DATA_PATH = os.path.join(DEFAULT_DATA_PATH, "nlu")

FINGERPRINT_FILE = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_DOMAIN_KEY = "domain"
FINGERPRINT_NLU_VERSION_KEY = "nlu_version"
FINGERPRINT_CORE_VERSION_KEY = "core_version"
FINGERPRINT_RASA_VERSION_KEY = "version"
FINGERPRINT_STORIES_KEY = "stories"
FINGERPRINT_MESSAGES_KEY = "messages"


def get_model(model_path, subdirectories=False):
    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)

    return unpack_model(model_path, subdirectories=subdirectories)


def unpack_model(model_file, working_directory=None, subdirectories=False):
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


def get_latest_model(model_path):
    if not os.path.exists(model_path) or os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)

    list_of_files = get_models(model_path)

    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)


def get_models(model_path):
    return glob.glob(os.path.join(model_path, "*.tar"))


def create_package_rasa(training_directory, model_directory, output_filename,
                        fingerprint=None):
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


def model_fingerprint(config_file, domain_file=None, nlu_data=None, stories=None):
    fingerprint = {
        FINGERPRINT_CONFIG_KEY: get_file_hash(config_file),
        FINGERPRINT_NLU_VERSION_KEY: rasa_nlu.__version__,
        FINGERPRINT_CORE_VERSION_KEY: rasa_core.__version__,
        FINGERPRINT_RASA_VERSION_KEY: rasa.__version__
    }

    if domain_file and os.path.isfile(domain_file):
        fingerprint[FINGERPRINT_DOMAIN_KEY] = get_file_hash(domain_file)

    if nlu_data and os.path.isdir(nlu_data):
        fingerprint[FINGERPRINT_MESSAGES_KEY] = [
            get_file_hash(os.path.join(nlu_data, f))
            for f in os.listdir(nlu_data)
            if not f.startswith('.')]
    elif nlu_data:
        fingerprint[FINGERPRINT_MESSAGES_KEY] = [get_file_hash(nlu_data)]

    if stories and os.path.isdir(stories):
        fingerprint[FINGERPRINT_STORIES_KEY] = [
            get_file_hash(os.path.join(stories, f))
            for f in os.listdir(stories)
            if not f.startswith('.')]
    elif stories:
        fingerprint[FINGERPRINT_STORIES_KEY] = [get_file_hash(stories)]

    return fingerprint


def fingerprint_from_path(model_path):
    fingerprint_path = os.path.join(model_path, FINGERPRINT_FILE)

    if os.path.isfile(fingerprint_path):
        return rasa_core.utils.read_json_file(fingerprint_path)
    else:
        return {}


def persist_fingerprint(path, fingerprint):
    path = os.path.join(path, FINGERPRINT_FILE)
    dump_obj_as_json_to_file(path, fingerprint)


def core_fingerprint_changed(fingerprint1, fingerprint2):
    relevant_keys = [FINGERPRINT_CONFIG_KEY, FINGERPRINT_CORE_VERSION_KEY,
                     FINGERPRINT_DOMAIN_KEY, FINGERPRINT_RASA_VERSION_KEY,
                     FINGERPRINT_STORIES_KEY]

    return any(
        [fingerprint1.get(k) != fingerprint2.get(k) for k in relevant_keys])


def nlu_fingerprint_changed(fingerprint1, fingerprint2):
    relevant_keys = [FINGERPRINT_CONFIG_KEY, FINGERPRINT_NLU_VERSION_KEY,
                     FINGERPRINT_RASA_VERSION_KEY, FINGERPRINT_MESSAGES_KEY]

    return any(
        [fingerprint1.get(k) != fingerprint2.get(k) for k in relevant_keys])


def merge_model(source: Text, target: Text) -> bool:
    try:
        shutil.move(source, target)
        return False
    except Exception as e:
        logging.debug(e)
        return True

