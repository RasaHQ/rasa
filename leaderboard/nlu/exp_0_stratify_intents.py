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
    exclusion_fraction: float = 0.0
    exclusion_seed: int = 345
    test_fraction: float = 0.2
    test_seed: int = 42

    @staticmethod
    def to_pattern() -> None:
        return ",".join(
            [
                "model:${model.name}",
                "data:${data.name}",
                "drop:${drop_intents_with_less_than}",
                "exclude:${exclusion_fraction}",
                "test:${test_fraction}",
                "test_seed:${test_seed}",
                "exclusion_seed:${exclusion_seed}",
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

    def _split(
        self, nlu_data: TrainingData, test_fraction: float, random_seed: int
    ) -> Tuple[TrainingData, TrainingData]:
        """Split given data into train and test."""

        data_messages = nlu_data.intent_examples
        data_indices = np.arange(len(data_messages))
        labels_str = [message.get("intent") for message in data_messages]

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_fraction,
            random_state=random_seed,
        )

        indices_per_split = next(sss.split(data_indices, labels_str))

        splits = []
        for split_name, indices in zip(["train", "test"], indices_per_split):
            split_messages = [data_messages[data_indices[idx]] for idx in indices]
            split_responses = nlu_data._needed_responses_for_examples(split_messages)
            split_data = TrainingData(
                split_messages,
                entity_synonyms=nlu_data.entity_synonyms,
                regex_features=nlu_data.regex_features,
                lookup_tables=nlu_data.lookup_tables,
                responses=split_responses,
            )
            splits.append(split_data)

        return splits

    def split(self, nlu_data: TrainingData) -> Tuple[TrainingData, TrainingData]:

        print(self.config)
        train, test = self._split(
            nlu_data=nlu_data,
            test_fraction=self.config.test_fraction,
            random_seed=self.config.test_seed,
        )
        if self.config.exclusion_fraction > 0:
            train, _ = self._split(
                nlu_data=train,
                test_fraction=self.config.exclusion_fraction,
                random_seed=self.config.exclusion_seed,
            )
        return train, test


if __name__ == "__main__":
    get_runner(config_class=Config, experiment_class=IntentExperiment)()
