import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from omegaconf import MISSING
import copy

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
    exclusion_ignore_leq: int = 10
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
                "exclusion_ignore_leq:${exclusion_ignore_leq}",
                "exclusion_seed:${exclusion_seed}",
            ]
        )


@dataclass
class EntityExperiment(BaseNLUExperiment):
    """Intent experiment using intent names only for stratification.

    Notes:
    - similar to exp_2 but using test data from a separate file
    - difference is that stratified sampling is only used to exclude points from the
      training data
    """

    def _pseudo_labels(self, nlu_data: TrainingData):
        """Use the (globally) most infrequent annotation as label."""
        # count how often each entity appears - and filter
        # entity counts according to the minimum count given
        # by the `exclusion_ignore_leq` parameter
        counts = copy.deepcopy(nlu_data.number_of_examples_per_entity)
        counts = {
            entity[len("entity '") : -1]: count
            for entity, count in counts.items()
            if count > self.config.exclusion_ignore_leq
        }
        # for each message, return the entity with the smallest
        # count (if that count is >= the `exclusion_ignore_leq`
        # parameter, or '' otherwise)
        filtered_entities_per_message = [
            [
                entity["entity"]
                for entity in message.get("entities", [])
                if str(entity["entity"]) in counts
            ]
            for message in nlu_data.nlu_examples
        ]
        return [
            min(
                entities,
                key=lambda entity_entity: counts[str(entity_entity)],
            )
            if entities
            else ""
            for entities in filtered_entities_per_message
        ]

    def exclude_data(self, nlu_data: TrainingData) -> TrainingData:
        labels = self._pseudo_labels(nlu_data)
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
    get_runner(config_class=Config, experiment_class=EntityExperiment)()
