import copy
import logging
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional

from leaderboard.nlu.base_nlu_experiment import BaseNLUExperiment
from leaderboard.utils import rasa_utils
from leaderboard.utils.experiment import ExperimentConfiguration, get_runner
from leaderboard.utils.split_utils import stratified_split_for_nlu_data
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__file__)


@dataclass
class Config(ExperimentConfiguration):
    remove_data_for_entities_with_num_examples_below: int = 0
    train_exclusion_fraction: float = 0.0
    exclusion_seed: int = 345
    test_fraction: float = 0.2
    test_seed: int = 42

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
                "test:${test_fraction}",
                "test_seed:${test_seed}",
                "exclusion_seed:${exclusion_seed}",
            ]
        )


@dataclass
class Experiment(BaseNLUExperiment):
    """Intent experiment using entity type combinations as labels for stratification.

    Notes:
    - see: _pseudo_labels
    """

    counts: Dict[str, int] = field(default_factory=dict)

    def preprocess(self, nlu_data: TrainingData) -> TrainingData:
        """Remove entities with too few examples."""
        self.counts = copy.deepcopy(nlu_data.number_of_examples_per_entity)
        # WHO DOES THAT?! ... WHY?! ...
        self.counts = {entity[len("entity '"):-1] : count for entity, count in
                       self.counts.items()}
        messages_with_filtered_entities = [
            Message(
                data={
                    "text": message.get("text"),
                    "intent": message.get("intent"),
                    "entities": [
                        entity
                        for entity in message.get("entities")
                        if self.counts.get(entity["entity"], 0)
                        >= self.config.remove_data_for_entities_with_num_examples_below
                    ],
                }
            )
            for message in nlu_data.nlu_examples
        ]
        return TrainingData(training_examples=messages_with_filtered_entities)

    def _pseudo_labels(self, nlu_data: TrainingData):
        """Use the (globally) most infrequent annotation as label."""
        return [
            min(
                [
                    (entity["entity"], self.counts[str(entity["entity"])]) # !!!
                    for entity in message.get("entities")
                ],
                key=lambda item: item[1],
            )[0]
            if message.get("entities")
            else ""
            for message in nlu_data.nlu_examples
        ]

    def split(
        self, nlu_data: TrainingData
    ) -> Tuple[TrainingData, Optional[TrainingData]]:

        pseudo_labels = self._pseudo_labels(nlu_data=nlu_data)

        (train, test), (train_indices, _) = stratified_split_for_nlu_data(
            nlu_data=nlu_data,
            test_fraction=self.config.test_fraction,
            random_seed=self.config.test_seed,
            labels=pseudo_labels,
        )
        if self.config.train_exclusion_fraction > 0:
            pseudo_labels_pruned = [pseudo_labels[idx] for idx in train_indices]
            (train, _), _ = stratified_split_for_nlu_data(
                nlu_data=train,
                test_fraction=self.config.train_exclusion_fraction,
                random_seed=self.config.exclusion_seed,
                labels=pseudo_labels_pruned,
            )
        return train, test

    def data_to_train_test_split(
        self, data: TrainingData
    ) -> Tuple[TrainingData, Optional[TrainingData]]:
        data = self.preprocess(data)
        return self.split(data)

    def load_train_test_split(self) -> Tuple[TrainingData, TrainingData]:
        """Preprocess and then split the data."""
        nlu_data = rasa_utils.load_nlu_data(data_path=self.config.data.data_path)
        return self.data_to_train_test_split(nlu_data)


if __name__ == "__main__":
    get_runner(config_class=Config, experiment_class=Experiment)()
