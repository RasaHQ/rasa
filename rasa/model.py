import copy
import glob
import hashlib
import logging
import os
import shutil
from subprocess import CalledProcessError, DEVNULL, check_output  # skipcq:BAN-B404
import tempfile
import typing
from pathlib import Path
from typing import Any, Text, Tuple, Union, Optional, List, Dict, NamedTuple

from packaging import version

from rasa.constants import MINIMUM_COMPATIBLE_VERSION
import rasa.shared.utils.io
import rasa.utils.io
from rasa.cli.utils import create_output_path
from rasa.shared.utils.cli import print_success
from rasa.shared.constants import (
    CONFIG_KEYS_CORE,
    CONFIG_KEYS_NLU,
    CONFIG_KEYS,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_MODELS_PATH,
    DEFAULT_CORE_SUBDIRECTORY_NAME,
    DEFAULT_NLU_SUBDIRECTORY_NAME,
)

from rasa.exceptions import ModelNotFound
from rasa.utils.common import TempDirectoryPath

if typing.TYPE_CHECKING:
    from rasa.shared.importers.importer import TrainingDataImporter

logger = logging.getLogger(__name__)


# Type alias for the fingerprint
Fingerprint = Dict[Text, Union[Optional[Text], List[Text], int, float]]

FINGERPRINT_FILE_PATH = "fingerprint.json"

FINGERPRINT_CONFIG_KEY = "config"
FINGERPRINT_CONFIG_CORE_KEY = "core-config"
FINGERPRINT_CONFIG_NLU_KEY = "nlu-config"
FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY = "config-without-epochs"
FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY = "domain"
FINGERPRINT_NLG_KEY = "nlg"
FINGERPRINT_RASA_VERSION_KEY = "version"
FINGERPRINT_STORIES_KEY = "stories"
FINGERPRINT_NLU_DATA_KEY = "messages"
FINGERPRINT_NLU_LABELS_KEY = "nlu_labels"
FINGERPRINT_PROJECT = "project"
FINGERPRINT_TRAINED_AT_KEY = "trained_at"


class Section(NamedTuple):
    """Specifies which fingerprint keys decide whether this sub-model is retrained."""

    name: Text
    relevant_keys: List[Text]


SECTION_CORE = Section(
    name="Core model",
    relevant_keys=[
        FINGERPRINT_CONFIG_KEY,
        FINGERPRINT_CONFIG_CORE_KEY,
        FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY,
        FINGERPRINT_STORIES_KEY,
        FINGERPRINT_RASA_VERSION_KEY,
    ],
)
SECTION_NLU = Section(
    name="NLU model",
    relevant_keys=[
        FINGERPRINT_CONFIG_KEY,
        FINGERPRINT_CONFIG_NLU_KEY,
        FINGERPRINT_NLU_DATA_KEY,
        FINGERPRINT_RASA_VERSION_KEY,
    ],
)
SECTION_NLG = Section(name="NLG responses", relevant_keys=[FINGERPRINT_NLG_KEY])


class FingerprintComparisonResult:
    """Container for the results of a fingerprint comparison."""

    def __init__(
        self,
        nlu: bool = True,
        core: bool = True,
        nlg: bool = True,
        force_training: bool = False,
    ):
        """Creates a `FingerprintComparisonResult` instance.

        Args:
            nlu: `True` if the NLU model should be retrained.
            core: `True` if the Core model should be retrained.
            nlg: `True` if the responses in the domain should be updated.
            force_training: `True` if a training of all parts is forced.
        """
        self.nlu = nlu
        self.core = core
        self.nlg = nlg
        self.force_training = force_training

    def is_training_required(self) -> bool:
        """Check if anything has to be retrained."""

        return any([self.nlg, self.nlu, self.core, self.force_training])

    def should_retrain_core(self) -> bool:
        """Check if the Core model has to be updated."""

        return self.force_training or self.core

    def should_retrain_nlg(self) -> bool:
        """Check if the responses have to be updated."""

        return self.should_retrain_core() or self.nlg

    def should_retrain_nlu(self) -> bool:
        """Check if the NLU model has to be updated."""

        return self.force_training or self.nlu


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
    from tarsafe import TarSafe

    if working_directory is None:
        working_directory = tempfile.mkdtemp()

    # All files are in a subdirectory.
    try:
        with TarSafe.open(model_file, mode="r:gz") as tar:
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


def create_package_rasa(
    training_directory: Text,
    output_filename: Text,
    fingerprint: Optional[Fingerprint] = None,
) -> Text:
    """Create a zipped Rasa model from trained model files.

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


async def model_fingerprint(file_importer: "TrainingDataImporter") -> Fingerprint:
    """Create a model fingerprint from its used configuration and training data.

    Args:
        file_importer: File importer which provides the training data and model config.

    Returns:
        The fingerprint.

    """
    import time

    config = await file_importer.get_config()
    domain = await file_importer.get_domain()
    stories = await file_importer.get_stories()
    nlu_data = await file_importer.get_nlu_data()

    responses = domain.responses

    # Do a copy of the domain to not change the actual domain (shallow is enough)
    domain = copy.copy(domain)
    # don't include the response texts in the fingerprint.
    # Their fingerprint is separate.
    domain.responses = {}

    return {
        FINGERPRINT_CONFIG_KEY: _get_fingerprint_of_config(
            config, exclude_keys=CONFIG_KEYS
        ),
        FINGERPRINT_CONFIG_CORE_KEY: _get_fingerprint_of_config(
            config, include_keys=CONFIG_KEYS_CORE
        ),
        FINGERPRINT_CONFIG_NLU_KEY: _get_fingerprint_of_config(
            config, include_keys=CONFIG_KEYS_NLU
        ),
        FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY: (
            _get_fingerprint_of_config_without_epochs(config)
        ),
        FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY: domain.fingerprint(),
        FINGERPRINT_NLG_KEY: rasa.shared.utils.io.deep_container_fingerprint(responses),
        FINGERPRINT_PROJECT: project_fingerprint(),
        FINGERPRINT_NLU_DATA_KEY: nlu_data.fingerprint(),
        FINGERPRINT_NLU_LABELS_KEY: nlu_data.label_fingerprint(),
        FINGERPRINT_STORIES_KEY: stories.fingerprint(),
        FINGERPRINT_TRAINED_AT_KEY: time.time(),
        FINGERPRINT_RASA_VERSION_KEY: rasa.__version__,
    }


def _get_fingerprint_of_config(
    config: Optional[Dict[Text, Any]],
    include_keys: Optional[List[Text]] = None,
    exclude_keys: Optional[List[Text]] = None,
) -> Text:
    if not config:
        return ""

    keys = include_keys or list(filter(lambda k: k not in exclude_keys, config.keys()))

    sub_config = {k: config[k] for k in keys if k in config}

    return rasa.shared.utils.io.deep_container_fingerprint(sub_config)


def _get_fingerprint_of_config_without_epochs(
    config: Optional[Dict[Text, Any]],
) -> Text:
    if not config:
        return ""

    copied_config = copy.deepcopy(config)

    for key in ["pipeline", "policies"]:
        if copied_config.get(key):
            for p in copied_config[key]:
                if "epochs" in p:
                    del p["epochs"]

    return rasa.shared.utils.io.deep_container_fingerprint(copied_config)


def fingerprint_from_path(model_path: Text) -> Fingerprint:
    """Load a persisted fingerprint.

    Args:
        model_path: Path to directory containing the fingerprint.

    Returns:
        The fingerprint or an empty dict if no fingerprint was found.
    """
    if not model_path or not os.path.exists(model_path):
        return {}

    fingerprint_path = os.path.join(model_path, FINGERPRINT_FILE_PATH)

    if os.path.isfile(fingerprint_path):
        return rasa.shared.utils.io.read_json_file(fingerprint_path)
    else:
        return {}


def persist_fingerprint(output_path: Text, fingerprint: Fingerprint) -> None:
    """Persist a model fingerprint.

    Args:
        output_path: Directory in which the fingerprint should be saved.
        fingerprint: The fingerprint to be persisted.

    """

    path = os.path.join(output_path, FINGERPRINT_FILE_PATH)
    rasa.shared.utils.io.dump_obj_as_json_to_file(path, fingerprint)


def did_section_fingerprint_change(
    fingerprint1: Fingerprint, fingerprint2: Fingerprint, section: Section
) -> bool:
    """Check whether the fingerprint of a section has changed."""
    for k in section.relevant_keys:
        if fingerprint1.get(k) != fingerprint2.get(k):
            logger.info(f"Data ({k}) for {section.name} section changed.")
            return True
    return False


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


def should_retrain(
    new_fingerprint: Fingerprint,
    old_model: Optional[Text],
    train_path: Text,
    has_e2e_examples: bool = False,
    force_training: bool = False,
) -> FingerprintComparisonResult:
    """Check which components of a model should be retrained.

    Args:
        new_fingerprint: The fingerprint of the new model to be trained.
        old_model: Path to the old zipped model file.
        train_path: Path to the directory in which the new model will be trained.
        has_e2e_examples: Whether the new training data contains e2e examples.
        force_training: Indicates if the model needs to be retrained even if the data
            has not changed.

    Returns:
        A FingerprintComparisonResult object indicating whether Rasa Core and/or Rasa
        NLU needs to be retrained or not.
    """
    fingerprint_comparison = FingerprintComparisonResult()

    if old_model is None or not os.path.exists(old_model):
        return fingerprint_comparison

    try:
        with unpack_model(old_model) as unpacked:
            last_fingerprint = fingerprint_from_path(unpacked)
            old_core, old_nlu = get_model_subdirectories(unpacked)

            fingerprint_comparison = FingerprintComparisonResult(
                core=did_section_fingerprint_change(
                    last_fingerprint, new_fingerprint, SECTION_CORE
                ),
                nlu=did_section_fingerprint_change(
                    last_fingerprint, new_fingerprint, SECTION_NLU
                ),
                nlg=did_section_fingerprint_change(
                    last_fingerprint, new_fingerprint, SECTION_NLG
                ),
                force_training=force_training,
            )

            # We should retrain core if nlu data changes and there are e2e stories.
            if has_e2e_examples and fingerprint_comparison.should_retrain_nlu():
                fingerprint_comparison.core = True

            core_merge_failed = False
            if not fingerprint_comparison.should_retrain_core():
                target_path = os.path.join(train_path, DEFAULT_CORE_SUBDIRECTORY_NAME)
                core_merge_failed = not move_model(old_core, target_path)
                fingerprint_comparison.core = core_merge_failed

            if not fingerprint_comparison.should_retrain_nlg() and core_merge_failed:
                # If moving the Core model failed, we should also retrain NLG
                fingerprint_comparison.nlg = True

            if not fingerprint_comparison.should_retrain_nlu():
                target_path = os.path.join(train_path, "nlu")
                fingerprint_comparison.nlu = not move_model(old_nlu, target_path)

            return fingerprint_comparison
    except Exception as e:
        logger.error(
            f"Failed to get the fingerprint. Error: {e}.\n"
            f"Proceeding with running default retrain..."
        )
        return fingerprint_comparison


def can_finetune(
    last_fingerprint: Fingerprint,
    new_fingerprint: Fingerprint,
    core: bool = False,
    nlu: bool = False,
) -> bool:
    """Checks if components of a model can be finetuned with incremental training.

    Args:
        last_fingerprint: The fingerprint of the old model to potentially be fine-tuned.
        new_fingerprint: The fingerprint of the new model.
        core: Check sections for finetuning a core model.
        nlu: Check sections for finetuning an nlu model.

    Returns:
        `True` if the old model can be finetuned, `False` otherwise.
    """
    section_keys = [
        FINGERPRINT_CONFIG_WITHOUT_EPOCHS_KEY,
    ]
    if core:
        section_keys.append(FINGERPRINT_DOMAIN_WITHOUT_NLG_KEY)
    if nlu:
        section_keys.append(FINGERPRINT_NLU_LABELS_KEY)

    fingerprint_changed = did_section_fingerprint_change(
        last_fingerprint,
        new_fingerprint,
        Section(name="finetune", relevant_keys=section_keys),
    )

    old_model_above_min_version = version.parse(
        last_fingerprint.get(FINGERPRINT_RASA_VERSION_KEY)
    ) >= version.parse(MINIMUM_COMPATIBLE_VERSION)
    return old_model_above_min_version and not fingerprint_changed


def package_model(
    fingerprint: Fingerprint,
    output_directory: Text,
    train_path: Text,
    fixed_model_name: Optional[Text] = None,
    model_prefix: Text = "",
) -> Text:
    """
    Compress a trained model.

    Args:
        fingerprint: fingerprint of the model
        output_directory: path to the directory in which the model should be stored
        train_path: path to uncompressed model
        fixed_model_name: name of the compressed model file
        model_prefix: prefix of the compressed model file

    Returns: path to 'tar.gz' model file
    """
    output_directory = create_output_path(
        output_directory, prefix=model_prefix, fixed_name=fixed_model_name
    )
    create_package_rasa(train_path, output_directory, fingerprint)

    print_success(
        "Your Rasa model is trained and saved at '{}'.".format(
            os.path.abspath(output_directory)
        )
    )

    return output_directory


async def update_model_with_new_domain(
    importer: "TrainingDataImporter", unpacked_model_path: Union[Path, Text]
) -> None:
    """Overwrites the domain of an unpacked model with a new domain.

    Args:
        importer: Importer which provides the new domain.
        unpacked_model_path: Path to the unpacked model.
    """
    model_path = Path(unpacked_model_path) / DEFAULT_CORE_SUBDIRECTORY_NAME
    domain = await importer.get_domain()
    domain.persist(model_path / DEFAULT_DOMAIN_PATH)


def get_model_for_finetuning(
    previous_model_file: Optional[Union[Path, Text]]
) -> Optional[Text]:
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
