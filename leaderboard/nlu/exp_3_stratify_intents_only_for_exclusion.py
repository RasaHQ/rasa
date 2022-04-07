import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from omegaconf import MISSING

from leaderboard.nlu.base_nlu_experiment import (
    BaseNLUExperiment,
)
from leaderboard.utils import rasa_utils
from leaderboard.utils.experiment import ExperimentConfiguration, get_runner
from leaderboard.utils.split_utils import stratified_split_for_nlu_data
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__file__)


@dataclass
class Config(ExperimentConfiguration):
    train_exclusion_fraction: float = 0.0
    exclusion_seed: int = 345
    test_data_path: str = MISSING

    @staticmethod
    def to_pattern() -> str:
        """For nicer experiment run folder names.

        It is not necessary to define all hyperparameters in the folder name.
        We will log (and read) the full experiment configuration from the run's
        `.hydra/config.yaml` file later.
        """
        return ",".join(
            [
                "model:${model.name}",
                "data:${data.name}",
                "exclude:${train_exclusion_fraction}",
                "exclusion_seed:${exclusion_seed}",
            ]
        )


@dataclass
class IntentExperiment(BaseNLUExperiment):
    """Intent experiment using intent names only for stratification.

    Notes:
    - similar to exp_0 but using test data from a separate file
    - differences are:
        1. stratified sampling is only used to exclude points from the
           training data
        2. no preprocessing which removes messages for infrequent intents
    """

    def exclude_data(self, nlu_data: TrainingData) -> TrainingData:
        labels = [message.get("intent") for message in nlu_data.nlu_examples]
        if self.config.train_exclusion_fraction > 0:
            (nlu_data, _), _ = stratified_split_for_nlu_data(
                nlu_data=nlu_data,
                labels=labels,
                test_fraction=self.config.train_exclusion_fraction,
                random_seed=self.config.exclusion_seed,
            )
        return nlu_data

    def load_train_test_split(self) -> Tuple[TrainingData, Optional[TrainingData]]:
        """Preprocess and then split the data."""
        train_data = rasa_utils.load_nlu_data(data_path=self.config.data.data_path)
        train_data = self.exclude_data(train_data)
        test_data = rasa_utils.load_nlu_data(data_path=self.config.test_data_path)
        return train_data, test_data


if __name__ == "__main__":
    get_runner(config_class=Config, experiment_class=IntentExperiment)()
