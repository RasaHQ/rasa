import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import hydra
from omegaconf import MISSING

import rasa.shared.utils.io as rasa_io_utils
from rasa.shared.constants import CONFIG_SCHEMA_FILE
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils import validation

logger = logging.getLogger(__name__)


def absolute_path(relative_path: Path) -> Path:
    """Use hydra to convert a relative to an absolute path."""
    return Path(hydra.utils.to_absolute_path(relative_path)).resolve()


@dataclass
class ModelConfig:
    """Basic Rasa model configuration needed for our experiments."""

    name: str = MISSING
    config_path: str = MISSING

    def validate(self) -> None:
        if self.config_path is not MISSING:
            # run Rasa's configuration validation
            validation.validate_yaml_schema(
                yaml_file_content=rasa_io_utils.read_file(
                    absolute_path(self.config_path)
                ),
                schema_path=CONFIG_SCHEMA_FILE,
            )


@dataclass
class DataConfig:
    """Basic Rasa data configuration needed for our experiments."""

    name: str = MISSING
    domain_path: str = MISSING
    data_path: str = MISSING

    def validate(self) -> None:
        for path in [self.domain_path, self.data_path]:
            if path is not MISSING:
                assert Path(path).is_file(), f"File {path} not found"


@dataclass
class ExperimentConfiguration:
    """Configuration of a basic intent classifier experiment."""

    description: str = "Basic Intent Experiment"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    clear_rasa_cache: bool = True

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()


@dataclass
class BaseExperiment(ABC):
    """Base Experiment."""

    @abstractmethod
    def get_description(self) -> str:
        ...

    @abstractmethod
    def load_splits(self) -> Tuple[TrainingData, TrainingData]:
        ...

    @abstractmethod
    def run(self, train: TrainingData, test: Optional[TrainingData]) -> None:
        ...

    def execute(self) -> None:
        # TODO: extend logging of which script is running
        logger.info(f"Running Experiment {self.__class__.__name__}")
        logger.info(f"Description: {self.get_description()}")
        train, test = self.load_splits()
        self.run(train, test)
