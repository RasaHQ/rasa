import logging
from typing import Dict, List, Optional, Text, Union

from rasa import data
from rasa.core.domain import Domain, InvalidDomain
from rasa.core.training.structures import StoryGraph
from rasa.importers import utils, autoconfig
from rasa.importers.importer import TrainingDataImporter
from rasa.importers.autoconfig import TrainingType
from rasa.nlu.training_data import TrainingData
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)


class RasaFileImporter(TrainingDataImporter):
    """Default `TrainingFileImporter` implementation."""

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        training_type: Optional[TrainingType] = TrainingType.BOTH,
    ):

        self._domain_path = domain_path

        self._nlu_files = data.get_data_files(training_data_paths, data.is_nlu_file)
        self._story_files = data.get_data_files(training_data_paths, data.is_story_file)

        self.config = autoconfig.get_configuration(config_file, training_type)

    async def get_config(self) -> Dict:
        return self.config

    async def get_stories(
        self,
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:

        return await utils.story_graph_from_paths(
            self._story_files,
            await self.get_domain(),
            template_variables,
            use_e2e,
            exclusion_percentage,
        )

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
