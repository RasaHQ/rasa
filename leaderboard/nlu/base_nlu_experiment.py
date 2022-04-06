import asyncio
import logging
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import rasa.model_training as model_training
import rasa.shared.utils.io as rasa_io_utils
from leaderboard.utils import rasa_utils
from leaderboard.utils.base_experiment import (
    absolute_path,
    BaseExperiment,
    ExperimentConfiguration,
)
from rasa.core.agent import Agent
from rasa.nlu.test import run_evaluation
from rasa.shared.nlu.training_data.training_data import TrainingData

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
        return rasa_utils.load_nlu_data(
            data_path=self.config.data.data_path,
            domain_path=self.config.data.domain_path,
        )

    @abstractmethod
    def preprocess(self, nlu_data: TrainingData) -> TrainingData:
        """Preprocessing applied to **all** data points, i.e. the loaded data."""
        ...

    @abstractmethod
    def split(
        self, nlu_data: TrainingData
    ) -> Tuple[TrainingData, Optional[TrainingData]]:
        """Split given data points into a train and test split."""
        ...

    def to_train_test_split(
        self, nlu_data: TrainingData
    ) -> Tuple[TrainingData, Optional[TrainingData]]:
        """Preprocess and then split the data.

        Yes, you could just override this method. But it is easier to follow what
        happens in an experiment if we keep preprocessing and splitting separate.
        """
        data = self.preprocess(nlu_data)
        return self.split(data)

    def run(self, train: TrainingData, test: Optional[TrainingData]) -> None:

        # create sub-folders
        data_path = self.out_dir / "data"
        data_path.mkdir(parents=True)
        report_path = self.out_dir / "report"
        report_path.mkdir()

        # save train data
        train_split_path = data_path / "train.yml"
        rasa_io_utils.write_text_file(train.nlu_as_yaml(), train_split_path)

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

        if test is not None:

            # save test data
            test_split_path = data_path / "test.yml"
            rasa_io_utils.write_text_file(test.nlu_as_yaml(), test_split_path)

            # evaluate
            processor = Agent.load(model_path=model_path).processor
            _ = asyncio.run(
                run_evaluation(
                    test_split_path,
                    processor,
                    output_directory=report_path,
                    errors=True,
                )
            )

        # save some stats on the data
        rasa_utils.extract_nlu_stats(train=train, test=test, report_path=report_path)

        if self.config.clear_rasa_cache:
            shutil.rmtree(self.out_dir / ".rasa")
