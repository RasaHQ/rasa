import logging
import os
import subprocess
from abc import abstractmethod, ABC
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional, Tuple, Type, Callable, List
import inspect

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from tqdm import tqdm

import rasa.shared.utils.io as rasa_io_utils
from leaderboard.utils import hydra_utils
from rasa.shared.constants import CONFIG_SCHEMA_FILE
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils import validation

logger = logging.getLogger(__name__)

HYDRA_CONFIG = "config"


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

    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    clear_rasa_cache: bool = True

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()

    @staticmethod
    def to_pattern() -> None:
        return "model:${model.name},data:${data.name}"


@dataclass
class BaseExperiment(ABC):
    """Base Experiment."""

    @abstractmethod
    def load_train_test(self) -> Tuple[TrainingData, TrainingData]:
        ...

    @abstractmethod
    def run(self, train: TrainingData, test: Optional[TrainingData]) -> None:
        ...

    def execute(self) -> None:
        script = inspect.getmodule(self.__class__).__file__
        logger.info(f"Running: {script}")
        train, test = self.load_train_test()
        self.run(train, test)


def get_runner(
    config_class: Type[ExperimentConfiguration], experiment_class: Type[BaseExperiment]
) -> Callable:
    """Registers config with hydra and returns a

    Args:
        config_class:
        experiment_class: an experiment class
    """
    ConfigStore.instance().store(name=HYDRA_CONFIG, node=config_class)

    @hydra.main(config_path=None, config_name=HYDRA_CONFIG)
    def main(config: ExperimentConfiguration) -> float:
        out_dir = Path.cwd()  # with multi-run, the cwd is the experiment sub-folder
        experiment = experiment_class(out_dir=out_dir, config=config)
        try:
            experiment.execute()
        except Exception as e:
            logger.exception(f"An error {e} occurred during this experiment.")
        return 1.0  # needed for sweepers, replace with some true score for hypopt

    return main


def multirun(
    experiment_module: ModuleType,
    configs: List[ExperimentConfiguration],
    out_dir: Path,
    capture: bool = False,
) -> None:
    """Executes multiple experiments.

    Args:
        experiment_module: must contain a class `Config` with staticmethod
         `to_pattern` and the corresponding file must be an experiment script
        configs: list of configurations to be run iteratively
        out_dir: result directory used for all experiments
        capture: set to True to capture all stdout/stderr output from the
          experiment script runs
    """

    experiment_pattern = (
        experiment_module.Config.to_pattern() + "__${now:%Y-%m-%d_%H-%M-%S}"
    )

    script = experiment_module.__file__
    script_name = os.path.basename(script).replace(".py", "")
    full_pattern = os.path.join(
        out_dir,
        f"{script_name}" + "__${data.name}",
        experiment_pattern,
    )

    command = [
        "python",
        script,
        f"hydra.run.dir='{full_pattern}'",
    ]

    # validate all configs before starting anything...
    for config in configs:
        config.validate()

    captured = (
        dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if capture else {}
    )
    for config in tqdm(configs):
        args = hydra_utils.to_hydra_cli_args(OmegaConf.structured(config))
        result = subprocess.run(command + args, **captured)
        if result.returncode != 0:
            logger.error("Configuration {config} could not be evaluated.")
