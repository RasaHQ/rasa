from functools import reduce
from typing import Text, Optional, Union, List, Dict, Any
import logging
from rasa.data import data

from rasa.core.agent import Agent
from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.dsl import StoryFileReader
from rasa.core.training.structures import StoryStep
from rasa.nlu.model import Interpreter
from rasa.nlu.training_data import TrainingData, loading
import rasa.utils.io as io_utils
import rasa.utils.common as common_utils

logger = logging.getLogger(__name__)


class TrainingFileImporter:
    async def get_domain(self) -> Domain:
        pass

    async def get_stories(
        self,
        agent: Agent,
        interpreter: Interpreter = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> List[DialogueStateTracker]:
        pass

    async def get_config(self) -> Dict:
        pass

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        pass

    @staticmethod
    def load_from_config(
        config_path: Text, domain_path: Text, training_data_paths: List[Text]
    ) -> "TrainingFileImporter":
        config = io_utils.read_config_file(config_path)
        return TrainingFileImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths
        )

    @staticmethod
    def load_from_dict(
        config: Dict,
        config_path: Text,
        domain_path: Text,
        training_data_paths: List[Text],
    ) -> "TrainingFileImporter":
        importers = config.get("importers", [])
        importers = [
            TrainingFileImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]

        if not importers:
            importers = [
                SimpleFileImporter(config_path, domain_path, training_data_paths)
            ]

        return CombinedFileImporter(importers)

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Text,
        training_data_paths: List[Text],
    ) -> Optional["TrainingFileImporter"]:
        module_path = importer_config.pop("type")
        if module_path == SimpleFileImporter.__name__:
            importer_class = SimpleFileImporter
        else:
            try:
                importer_class = common_utils.class_from_module_path(module_path)
            except (AttributeError, ImportError):
                logging.warning("Importer '{}' not found.".format(module_path))
                return None

        return importer_class(
            config_path, domain_path, training_data_paths, **importer_config
        )


class CombinedFileImporter(TrainingFileImporter):
    def __init__(self, importers: List[TrainingFileImporter]):
        self._importers = importers

    async def get_config(self) -> Dict:
        configs = [await importer.get_config() for importer in self._importers]
        return reduce(lambda merged, other: {**merged, **other}, configs, {})

    async def get_domain(self) -> Domain:
        domains = [await importer.get_domain() for importer in self._importers]
        return reduce(
            lambda merged, other: merged.merge(other), domains, Domain.empty()
        )

    async def get_stories(
        self,
        agent: Agent,
        interpreter: Interpreter = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> List[StoryStep]:
        story_steps = [
            await importer.get_stories(
                agent, interpreter, template_variables, use_e2e, exclusion_percentage
            )
            for importer in self._importers
        ]
        return reduce(lambda merged, other: merged + other, story_steps, [])

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        nlu_datas = [
            await importer.get_nlu_data(language) for importer in self._importers
        ]
        return reduce(
            lambda merged, other: merged.merge(other), nlu_datas, TrainingData()
        )


class SimpleFileImporter(TrainingFileImporter):
    def __init__(
        self,
        config_file: Text,
        domain_path: Text,
        training_data_paths: Optional[Union[List[Text], Text]],
    ):
        self.config = io_utils.read_config_file(config_file)
        self.domain = Domain.load(domain_path)

        self._story_files, self._nlu_files = data.get_core_nlu_files(
            training_data_paths
        )

    async def get_config(self, **kwargs: Optional[Dict[Text, Any]]) -> Dict:
        return self.config

    async def get_stories(
        self,
        agent: Agent,
        interpreter: Interpreter = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> List[StoryStep]:

        return await StoryFileReader.read_from_files(
            self._story_files,
            await self.get_domain(),
            interpreter,
            template_variables,
            use_e2e,
            exclusion_percentage,
        )

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        training_datas = [
            loading.load_data(nlu_file, language) for nlu_file in self._nlu_files
        ]
        return TrainingData().merge(*training_datas)

    async def get_domain(self) -> Domain:
        return self.domain
