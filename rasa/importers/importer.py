import asyncio
from functools import reduce
from typing import Text, Optional, List, Dict
import logging

from rasa.core.domain import Domain
from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
from rasa.core.training.structures import StoryGraph
from rasa.nlu.training_data import TrainingData
import rasa.utils.io as io_utils
import rasa.utils.common as common_utils

logger = logging.getLogger(__name__)


class TrainingDataImporter:
    """Common interface for different mechanisms to load training data."""

    async def get_domain(self) -> Domain:
        """Retrieves the domain of the bot.

        Returns:
            Loaded ``Domain``.
        """
        raise NotImplementedError()

    async def get_stories(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        """Retrieves the stories that should be used for training.

        Args:
            interpreter: Interpreter that should be used to parse end to
                         end learning annotations.
            template_variables: Values of templates that should be replaced while
                                reading the story files.
            use_e2e: Specifies whether to parse end to end learning annotations.
            exclusion_percentage: Amount of training data that should be excluded.

        Returns:
            ``StoryGraph`` containing all loaded stories.
        """

        raise NotImplementedError()

    async def get_config(self) -> Dict:
        """Retrieves the configuration that should be used for the training.

        Returns:
            The configuration as dictionary.
        """

        raise NotImplementedError()

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves the NLU training data that should be used for training.

        Args:
            language: Can be used to only load training data for a certain language.

        Returns:
            Loaded NLU ``TrainingData``.
        """

        raise NotImplementedError()

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """Loads a ``TrainingDataImporter`` instance from a configuration file."""

        config = io_utils.read_config_file(config_path)
        return TrainingDataImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths
        )

    @staticmethod
    def load_core_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """Loads a ``TrainingDataImporter`` instance from a configuration file that
           only reads Core training data.
        """

        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths
        )

        return CoreDataImporter(importer)

    @staticmethod
    def load_nlu_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """Loads a ``TrainingDataImporter`` instance from a configuration file that
           only reads NLU training data.
        """

        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths
        )

        return NluDataImporter(importer)

    @staticmethod
    def load_from_dict(
        config: Optional[Dict],
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """Loads a ``TrainingDataImporter`` instance from a dictionary."""

        from rasa.importers.rasa import RasaFileImporter

        config = config or {}
        importers = config.get("importers", [])
        importers = [
            TrainingDataImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]

        if not importers:
            importers = [
                RasaFileImporter(config_path, domain_path, training_data_paths)
            ]

        return CombinedDataImporter(importers)

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> Optional["TrainingDataImporter"]:
        from rasa.importers.multi_project import MultiProjectImporter
        from rasa.importers.rasa import RasaFileImporter

        module_path = importer_config.pop("name", None)
        if module_path == RasaFileImporter.__name__:
            importer_class = RasaFileImporter
        elif module_path == MultiProjectImporter.__name__:
            importer_class = MultiProjectImporter
        else:
            try:
                importer_class = common_utils.class_from_module_path(module_path)
            except (AttributeError, ImportError):
                logging.warning("Importer '{}' not found.".format(module_path))
                return None

        constructor_arguments = common_utils.minimal_kwargs(
            importer_config, importer_class
        )
        return importer_class(
            config_path, domain_path, training_data_paths, **constructor_arguments
        )


class NluDataImporter(TrainingDataImporter):
    """Importer that skips any Core-related file reading."""

    def __init__(self, actual_importer: TrainingDataImporter):
        self._importer = actual_importer

    async def get_domain(self) -> Domain:
        return Domain.empty()

    async def get_stories(
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


class CoreDataImporter(TrainingDataImporter):
    """Importer that skips any NLU related file reading."""

    def __init__(self, actual_importer: TrainingDataImporter):
        self._importer = actual_importer

    async def get_domain(self) -> Domain:
        return await self._importer.get_domain()

    async def get_stories(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        return await self._importer.get_stories(
            interpreter, template_variables, use_e2e, exclusion_percentage
        )

    async def get_config(self) -> Dict:
        return await self._importer.get_config()

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        return TrainingData()


class CombinedDataImporter(TrainingDataImporter):
    """A ``TrainingDataImporter`` that supports using multiple ``TrainingDataImporter``s as
        if they were a single instance.
    """

    def __init__(self, importers: List[TrainingDataImporter]):
        self._importers = importers

    async def get_config(self) -> Dict:
        configs = [importer.get_config() for importer in self._importers]
        configs = await asyncio.gather(*configs)

        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    async def get_domain(self) -> Domain:
        domains = [importer.get_domain() for importer in self._importers]
        domains = await asyncio.gather(*domains)

        return reduce(
            lambda merged, other: merged.merge(other), domains, Domain.empty()
        )

    async def get_stories(
        self,
        interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
        template_variables: Optional[Dict] = None,
        use_e2e: bool = False,
        exclusion_percentage: Optional[int] = None,
    ) -> StoryGraph:
        stories = [
            importer.get_stories(
                interpreter, template_variables, use_e2e, exclusion_percentage
            )
            for importer in self._importers
        ]
        stories = await asyncio.gather(*stories)

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        nlu_data = [importer.get_nlu_data(language) for importer in self._importers]
        nlu_data = await asyncio.gather(*nlu_data)

        return reduce(
            lambda merged, other: merged.merge(other), nlu_data, TrainingData()
        )
