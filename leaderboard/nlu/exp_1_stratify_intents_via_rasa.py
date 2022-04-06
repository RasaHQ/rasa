import logging
from dataclasses import dataclass
from typing import Tuple

from leaderboard.utils.base_experiment import ExperimentConfiguration, get_runner
from leaderboard.nlu.base_nlu_experiment import BaseNLUExperiment
from rasa.nlu.test import drop_intents_below_freq
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__file__)


@dataclass
class Config(ExperimentConfiguration):
    remove_data_for_intents_with_num_examples_leq: int = 5
    train_exclusion_fraction: float = 0.0
    exclusion_seed: int = 345
    test_fraction: float = 0.2
    test_seed: int = 42

    @staticmethod
    def to_pattern() -> None:
        return ",".join(
            [
                "model:${model.name}",
                "data:${data.name}",
                "drop:${remove_data_for_intents_with_num_examples_leq}",
                "exclude:${train_exclusion_fraction}",
                "test:${test_fraction}",
                "test_seed:${test_seed}",
                "exclusion_seed:${exclusion_seed}",
            ]
        )


@dataclass
class IntentExperiment(BaseNLUExperiment):
    """Intent experiment using intent names + responses for stratification.

    Notes:
    - stratified splits computed based on intent names + responses (if present)
    - rasa's stratification method is used
    """

    def preprocess(self, nlu_data: TrainingData) -> TrainingData:
        """Preprocessing applied to **all** data points."""
        nlu_data = drop_intents_below_freq(
            nlu_data, cutoff=self.config.remove_data_for_intents_with_num_examples_leq
        )
        return nlu_data

    def split(self, nlu_data: TrainingData) -> Tuple[TrainingData, TrainingData]:

        train, test = nlu_data.train_test_split(
            train_frac=1.0 - self.config.test_fraction,
            random_seed=self.config.test_seed,
        )
        if self.config.train_exclusion_fraction > 0:
            train, _ = train.train_test_split(
                train_frac=1.0 - self.config.train_exclusion_fraction,
                random_seed=self.config.exclusion_seed,
            )
        return train, test


if __name__ == "__main__":
    get_runner(config_class=Config, experiment_class=IntentExperiment)()
