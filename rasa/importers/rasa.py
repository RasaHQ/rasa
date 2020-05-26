import logging
import os
from typing import Dict, List, Optional, Text, Union

from rasa import data
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.training.dsl import StoryFileReader
from rasa.core.training.structures import StoryGraph
from rasa.importers import utils
from rasa.importers.importer import TrainingDataImporter
from rasa.nlu.training_data import TrainingData
from rasa.utils import io as io_utils
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)


class RasaFileImporter(TrainingDataImporter):
    """Default `TrainingFileImporter` implementation."""

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
    ):
        if config_file and os.path.exists(config_file):
            self.config = io_utils.read_config_file(config_file)
        else:
            self.config = {}

        self._domain_path = domain_path

        self._story_files, self._nlu_files = data.get_core_nlu_files(
            training_data_paths
        )

    async def get_config(self) -> Dict:
        return self.config

    async def get_stories(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:

        story_steps = await StoryFileReader.read_from_files(
            self._story_files,
            await self.get_domain(),
            interpreter,
            template_variables,
            use_e2e,
            exclusion_percentage,
        )
        return StoryGraph(story_steps)

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        return utils.training_data_from_paths(self._nlu_files, language)

    async def get_domain(self) -> Domain:
        domain = Domain.empty()
        try:
            domain = Domain.load(self._domain_path)
            domain.check_missing_templates()
        except InvalidDomain as e:
            raise_warning(
                f"Loading domain from '{self._domain_path}' failed. Using "
                f"empty domain. Error: '{e.message}'"
            )

        return domain
