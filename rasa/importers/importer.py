from functools import reduce
from typing import Text, Optional, List, Dict
import logging

import rasa.utils.common
from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa.core.training.structures import StoryGraph
from rasa.importers.simple import SimpleFileImporter
from rasa.nlu.training_data import TrainingData
import rasa.utils.io as io_utils
import rasa.utils.common as common_utils

logger = logging.getLogger(__name__)


class TrainingFileImporter:
    """Common interface for different mechanisms to load training data."""

    async def get_domain(self) -> Domain:
        """Retrieves the domain which should be used for the training."""
        raise NotImplementedError()

    async def get_story_data(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        """Retrieves the story data which should be used for the training."""

        raise NotImplementedError()

    async def get_config(self) -> Dict:
        """Retrieves the configuration which should be used for the training."""
        raise NotImplementedError()

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves the nlu training data which should be used for the training."""
        raise NotImplementedError()

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        load_only_nlu_data: bool = False,
    ) -> "TrainingFileImporter":
        """Loads a `TrainingFileImporter` instance from a configuration file."""

        config = io_utils.read_config_file(config_path)
        return TrainingFileImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths, load_only_nlu_data
        )

    @staticmethod
    def load_from_dict(
        config: Optional[Dict],
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
        load_only_nlu_data: bool = False,
    ) -> "TrainingFileImporter":
        """Loads a `TrainingFileImporter` instance from a dictionary."""
        config = config or {}
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

        importer = CombinedFileImporter(importers)
        if load_only_nlu_data:
            importer = NluFileImporter(importer)

        return importer

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> Optional["TrainingFileImporter"]:
        from rasa.importers.skill import SkillSelector

        module_path = importer_config.pop("name", None)
        if module_path == SimpleFileImporter.__name__:
            importer_class = SimpleFileImporter
        elif module_path == SkillSelector.__name__:
            importer_class = SkillSelector
        else:
            try:
                importer_class = common_utils.class_from_module_path(module_path)
            except (AttributeError, ImportError):
                logging.warning("Importer '{}' not found.".format(module_path))
                return None

        import rasa.cli.utils as cli_utils

        constructor_arguments = rasa.utils.common.minimal_kwargs(
            importer_config, importer_class
        )
        return importer_class(
            config_path, domain_path, training_data_paths, **constructor_arguments
        )


class NluFileImporter(TrainingFileImporter):
    """Importer which skips any Core related file reading"""

    def __init__(self, actual_importer: TrainingFileImporter):
        self._importer = actual_importer

    async def get_domain(self) -> Domain:
        return Domain.empty()

    async def get_story_data(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        return StoryGraph([])

    async def get_config(self) -> Dict:
        return await self._importer.get_config()

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        return await self._importer.get_nlu_data(language)


class CombinedFileImporter(TrainingFileImporter):
    """`TrainingFileImporter` which supports using multiple `TrainingFileImporter` as
        if it would be a single instance.
    """

    def __init__(self, importers: List[TrainingFileImporter]):
        self._importers = importers

    async def get_config(self) -> Dict:
        configs = []
        # Do this in a loop because Python 3.5 does not support async comprehensions
        for importer in self._importers:
            configs.append(await importer.get_config())

        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    async def get_domain(self) -> Domain:
        domains = []
        for importer in self._importers:
            domains.append(await importer.get_domain())

        return reduce(
            lambda merged, other: merged.merge(other), domains, Domain.empty()
        )

    async def get_story_data(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        story_graphs = []
        # Do this in a loop because Python 3.5 does not support async comprehensions
        for importer in self._importers:
            graph = await importer.get_story_data(
                interpreter, template_variables, use_e2e, exclusion_percentage
            )
            story_graphs.append(graph)

        return reduce(
            lambda merged, other: merged.merge(other), story_graphs, StoryGraph([])
        )

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        nlu_datas = []
        # Do this in a loop because Python 3.5 does not support async comprehensions
        for importer in self._importers:
            nlu_data = await importer.get_nlu_data(language)
            nlu_datas.append(nlu_data)
        training_data = reduce(
            lambda merged, other: merged.merge(other), nlu_datas, TrainingData()
        )
        return training_data
