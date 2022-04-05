import asyncio
import logging
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import rasa.cli.utils as rasa_cli_utils
import rasa.model_training as model_training
import rasa.shared.utils.io as rasa_io_utils
from leaderboard.utils.base_experiment import (
    absolute_path,
    BaseExperiment,
    ExperimentConfiguration,
)
from leaderboard.utils import rasa_utils
from rasa.core.agent import Agent
from rasa.nlu.test import run_evaluation
from rasa.shared.constants import DEFAULT_DATA_PATH, CONFIG_SCHEMA_FILE
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils import validation

logger = logging.getLogger(__file__)


@dataclass
class BaseNLUExperiment(BaseExperiment):
    """Intent Experiment using Rasa's split function.

    Notes:
    - using rasa's split function
    - additionally helps to exclude a fixed percentage of examples via
      rasa's split function
    - shoud mimic rasa test (i.e. no rng is passed on, the split functions
      use the given seed to initiate an rng)
    """

    config: ExperimentConfiguration
    out_dir: Path

    def load_data(self) -> TrainingData:
        """Load data from disk."""
        data_path = absolute_path(self.config.data.data_path)
        domain_path = absolute_path(self.config.data.domain_path)
        data_path = rasa_cli_utils.get_validated_path(
            data_path, "nlu", DEFAULT_DATA_PATH
        )
        test_data_importer = TrainingDataImporter.load_from_dict(
            training_data_paths=[str(data_path)], domain_path=domain_path
        )
        nlu_data = test_data_importer.get_nlu_data()
        return nlu_data

    @abstractmethod
    def preprocess(self, nlu_data: TrainingData) -> TrainingData:
        ...

    @abstractmethod
    def split(self, nlu_data: TrainingData) -> Tuple[TrainingData, TrainingData]:
        ...

    def load_train_test(self) -> Tuple[TrainingData, TrainingData]:
        data = self.load_data()
        data = self.preprocess(data)
        return self.split(data)

    def run(self, train: TrainingData, test: Optional[TrainingData]) -> None:

        # as before, train and test used in the experiment are persisted
        data_path = self.out_dir / "data"
        data_path.mkdir(parents=True)
        report_path = self.out_dir / "report"
        report_path.mkdir()

        #  store training data
        train_split_path = data_path / "train.yml"
        rasa_io_utils.write_text_file(train.nlu_as_yaml(), train_split_path)
        # TODO log stats for training data

        # train model
        model_output_path = self.out_dir / "model"
        model_path = model_training.train_nlu(
            config=absolute_path(self.config.model.config_path),
            nlu_data=train_split_path,
            output=model_output_path,
            fixed_model_name=self.config.model.name,
        )

        # extract training meta data
        model_archive = model_output_path / (self.config.model.name + ".tar.gz")
        rasa_utils.extract_metadata(model_archive, report_path)

        # test model
        if test is not None:
            test_split_path = data_path / "test.yml"
            rasa_io_utils.write_text_file(test.nlu_as_yaml(), test_split_path)

            processor = Agent.load(model_path=model_path).processor
            _ = asyncio.run(
                run_evaluation(
                    test_split_path,
                    processor,
                    output_directory=report_path,
                    errors=True,
                )
            )

        if self.config.clear_rasa_cache:
            shutil.rmtree(self.out_dir / ".rasa")
