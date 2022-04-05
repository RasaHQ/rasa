import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from leaderboard.utils.base_experiment import ExperimentConfiguration, get_runner
from leaderboard.nlu.base_nlu_experiment import (
    BaseNLUExperiment,
)
from rasa.nlu.test import drop_intents_below_freq
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__file__)


@dataclass
class Config(ExperimentConfiguration):
    drop_intents_with_less_than: int = 5
    exclusion_percentage: float = 0.0
    test_fraction: float = 0.2
    random_seed: int = 42

    @staticmethod
    def to_pattern() -> None:
        return ",".join(
            [
                "model:${model.name}",
                "data:${data.name}",
                "drop:${drop_intents_with_less_than}",
                "exclude:${exclusion_percentage}",
                "test:${test_fraction}",
                "seed:${random_seed}",
            ]
        )

@dataclass
class IntentExperiment(BaseNLUExperiment):
    """Intent experiment using intent names only for stratification.

    Notes:
    - stratified splits computed based on intent names only - responses are ignored
    - i.e. intent prediction should be more difficult than in the usual rasa test
      (see also exp_1) where stratification takes the responses in to account
    """

    def preprocess(self, nlu_data: TrainingData) -> TrainingData:
        """Preprocessing applied to **all** data points."""
        nlu_data = drop_intents_below_freq(
            nlu_data, cutoff=self.config.drop_intents_with_less_than
        )
        return nlu_data

    def split(self, nlu_data: TrainingData) -> Tuple[TrainingData, TrainingData]:

        train, test = nlu_data.train_test_split(
            train_frac=1.-self.config.test_fraction,
                                    random_seed=self.config.random_seed)
        if self.config.exclusion_percentage > 0:
            train, _ = self._split(
                nlu_data=train,
                test_fraction=self.config.exclusion_percentage,
                random_seed=self.config.random_seed + 1,
            )
        return train, test

if __name__ == "__main__":
    get_runner(config_class=Config, experiment_class=IntentExperiment)()
