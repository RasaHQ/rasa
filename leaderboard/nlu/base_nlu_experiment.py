import asyncio
import logging
import os.path
import shutil
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import rasa.model_training as model_training
import rasa.shared.utils.io as rasa_io_utils
from leaderboard.utils import rasa_utils
from leaderboard.utils.experiment import (
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

    def run(self, train: TrainingData, test: Optional[TrainingData]) -> None:

        # create sub-folders
        data_path = self.out_dir / "data"
        report_path = self.out_dir / "report"
        model_path = self.out_dir / "model"
        for sub_dir in [data_path, report_path, model_path]:
            sub_dir.mkdir(parents=True)

        # save train data
        train_split_path = data_path / "train.yml"
        rasa_io_utils.write_text_file(train.nlu_as_yaml(), train_split_path)

        # save model config
        model_config_file_name = os.path.basename(self.config.model.config_path)
        shutil.copy(
            src=absolute_path(self.config.model.config_path),
            dst=model_path / model_config_file_name,
        )

        # TODO: double-check here whether random seeds are fixed in model config?

        # train model
        _ = model_training.train_nlu(
            config=absolute_path(self.config.model.config_path),
            nlu_data=train_split_path,
            output=model_path,
            fixed_model_name=self.config.model.name,
        )

        # extract training meta data
        model_archive = model_path / (self.config.model.name + ".tar.gz")
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
