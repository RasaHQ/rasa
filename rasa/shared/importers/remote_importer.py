import os
from typing import Dict, List, Optional, Text, Union

import structlog
from tarsafe import TarSafe

import rasa.shared.core.flows.yaml_flows_io
import rasa.shared.data
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.nlu.persistor import StorageType
from rasa.shared.core.domain import Domain, InvalidDomain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.exceptions import RasaException
from rasa.shared.importers import utils
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.yaml import read_model_configuration

structlogger = structlog.get_logger()

TRAINING_DATA_ARCHIVE = "training_data.tar.gz"


class RemoteTrainingDataImporter(TrainingDataImporter):
    """Remote `TrainingFileImporter` implementation.

    Fetches training data from a remote storage and extracts it to a local directory.
    Extracted training data is then used to load flows, NLU, stories,
        domain, and config files.
    """

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        project_directory: Optional[Text] = None,
        remote_storage: Optional[StorageType] = None,
        training_data_path: Optional[Text] = None,
    ):
        """Initializes `RemoteTrainingDataImporter`.

        Args:
            config_file: Path to the model configuration file.
            domain_path: Path to the domain file.
            training_data_paths: List of paths to the training data files.
            project_directory: Path to the project directory.
            remote_storage: Storage to use to load the training data.
            training_data_path: Path to the training data.
        """
        self.remote_storage = remote_storage
        self.training_data_path = training_data_path

        self.extracted_path = self._fetch_and_extract_training_archive(
            TRAINING_DATA_ARCHIVE, self.training_data_path
        )

        self._nlu_files = rasa.shared.data.get_data_files(
            self.extracted_path, rasa.shared.data.is_nlu_file
        )
        self._story_files = rasa.shared.data.get_data_files(
            self.extracted_path, YAMLStoryReader.is_stories_file
        )
        self._flow_files = rasa.shared.data.get_data_files(
            self.extracted_path, rasa.shared.core.flows.yaml_flows_io.is_flows_file
        )
        self._conversation_test_files = rasa.shared.data.get_data_files(
            self.extracted_path, YAMLStoryReader.is_test_stories_file
        )

        self.config_file = config_file

    def _fetch_training_archive(
        self, training_file: str, training_data_path: Optional[str] = None
    ) -> str:
        """Fetches training files from remote storage."""
        from rasa.nlu.persistor import get_persistor

        persistor = get_persistor(self.remote_storage)
        if persistor is None:
            raise RasaException(
                f"Could not find a persistor for "
                f"the storage type '{self.remote_storage}'."
            )

        return persistor.retrieve(training_file, training_data_path)

    def _fetch_and_extract_training_archive(
        self, training_file: str, training_data_path: Optional[Text] = None
    ) -> Optional[str]:
        """Fetches and extracts training files from remote storage.

        If the `training_data_path` is not provided, the training
        data is extracted to the current working directory.

        Args:
            training_file: Name of the training data archive file.
            training_data_path: Path to the training data.

        Returns:
            Path to the extracted training data.
        """

        if training_data_path is None:
            training_data_path = os.path.join(os.getcwd(), "data")

        if os.path.isfile(training_data_path):
            raise ValueError(
                f"Training data path '{training_data_path}' is a file. "
                f"Please provide a directory path."
            )

        structlogger.debug(
            "rasa.importers.remote_training_data_importer.fetch_training_archive",
            training_data_path=training_data_path,
        )
        training_archive_file_path = self._fetch_training_archive(
            training_file, training_data_path
        )

        if not os.path.isfile(training_archive_file_path):
            raise FileNotFoundError(
                f"Training data archive '{training_archive_file_path}' not found. "
                f"Please make sure to provide the correct path."
            )

        structlogger.debug(
            "rasa.importers.remote_training_data_importer.extract_training_archive",
            training_archive_file_path=training_archive_file_path,
            training_data_path=training_data_path,
        )
        with TarSafe.open(training_archive_file_path, "r:gz") as tar:
            tar.extractall(path=training_data_path)

        structlogger.debug(
            "rasa.importers.remote_training_data_importer.remove_downloaded_archive",
            training_data_path=training_data_path,
        )
        os.remove(training_archive_file_path)
        return training_data_path

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        if not self.config_file or not os.path.exists(self.config_file):
            structlogger.debug(
                "rasa.importers.remote_training_data_importer.no_config_file",
                message="No configuration file was provided to the RasaFileImporter.",
            )
            return {}

        config = read_model_configuration(self.config_file)
        return config

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self.config_file

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return utils.story_graph_from_paths(
            self._story_files, self.get_domain(), exclusion_percentage
        )

    def get_flows(self) -> FlowsList:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return utils.flows_from_paths(self._flow_files)

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return utils.story_graph_from_paths(
            self._conversation_test_files, self.get_domain()
        )

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves NLU training data (see parent class for full docstring)."""
        return utils.training_data_from_paths(self._nlu_files, language)

    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        domain = Domain.empty()
        domain_path = f"{self.extracted_path}"
        try:
            domain = Domain.load(domain_path)
        except InvalidDomain as e:
            rasa.shared.utils.io.raise_warning(
                f"Loading domain from '{domain_path}' failed. Using "
                f"empty domain. Error: '{e}'"
            )

        return domain
