from typing import Tuple, List, Optional
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit

from rasa.shared.nlu.training_data.training_data import TrainingData


def stratified_split_for_nlu_data(
    nlu_data: TrainingData, labels: List, test_fraction: float, random_seed: int
) -> Tuple[Tuple[TrainingData, Optional[TrainingData]], Tuple[np.ndarray, np.ndarray]]:
    """Split given data into train and test."""

    data_messages = nlu_data.intent_examples
    data_indices = np.arange(len(data_messages))

    if test_fraction <= 0:
        return (nlu_data, None), (data_indices, np.ndarray([]))

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_seed,
    )

    indices_per_split = next(sss.split(data_indices, labels))

    splits = []
    splits_indices = []
    for split_name, indices in zip(["train", "test"], indices_per_split):
        split_messages = [data_messages[idx] for idx in indices]
        split_responses = nlu_data._needed_responses_for_examples(split_messages)
        split_data = TrainingData(
            split_messages,
            entity_synonyms=nlu_data.entity_synonyms,
            regex_features=nlu_data.regex_features,
            lookup_tables=nlu_data.lookup_tables,
            responses=split_responses,
        )
        splits_indices.append(indices)
        splits.append(split_data)

    return splits, splits_indices
