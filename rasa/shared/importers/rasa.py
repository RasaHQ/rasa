import logging
import os
from typing import Dict, List, Optional, Text, Union

import rasa.shared.data
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers import utils
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)

logger = logging.getLogger(__name__)


class RasaFileImporter(TrainingDataImporter):
    """Default `TrainingFileImporter` implementation."""

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
    ):

        self._domain_path = domain_path

        self._nlu_files = rasa.shared.data.get_data_files(
            training_data_paths, rasa.shared.data.is_nlu_file
        )
        self._story_files = rasa.shared.data.get_data_files(
            training_data_paths, YAMLStoryReader.is_stories_file
        )
        self._conversation_test_files = rasa.shared.data.get_data_files(
            training_data_paths, YAMLStoryReader.is_test_stories_file
        )

        self.config_file = config_file

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        if not self.config_file or not os.path.exists(self.config_file):
            logger.debug("No configuration file was provided to the RasaFileImporter.")
            return {}

        config = rasa.shared.utils.io.read_model_configuration(self.config_file)
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

        # If domain path is None, return an empty domain
        if not self._domain_path:
            return domain
        try:
            domain = Domain.load(self._domain_path)
        except InvalidDomain as e:
            rasa.shared.utils.io.raise_warning(
                f"Loading domain from '{self._domain_path}' failed. Using "
                f"empty domain. Error: '{e}'"
            )

        return domain
