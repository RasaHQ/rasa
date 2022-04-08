import tempfile
from pathlib import Path

import pytest
import logging
from typing import List

from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from leaderboard.nlu.exp_3_stratify_intents_only_for_exclusion import (
    Config,
    IntentExperiment,
)

logger = logging.getLogger(__file__)


@pytest.mark.parametrize(
    "counts, exclude_fraction, expected_train",
    [
        ([100, 200, 4], 0.5, [50, 100, 2]),
        ([100, 200, 4], 0.1, [90, 180, 3]),
    ],
)
def test_to_train_test_split_without_responses(
    tmp_path: Path,
    counts: List[int],
    exclude_fraction: float,
    expected_train: List[int],
):
    messages = [
        Message(data={TEXT: f"{intent_idx}-{message_idx}", INTENT: f"{intent_idx}"})
        for intent_idx, intent_count in enumerate(counts)
        for message_idx in range(intent_count)
    ]
    nlu_data = TrainingData(messages)

    results_per_seed = []
    for seed in [1, 2]:

        experiment = IntentExperiment(
            config=Config(
                train_exclusion_fraction=exclude_fraction,
                exclusion_seed=seed,
                model=None,
                data=None,
            ),
            out_dir=tmp_path,
        )
        train = experiment.exclude_data(nlu_data=nlu_data)
        results_per_seed.append(train)

        for intent_idx, expected_count in enumerate(expected_train):
            assert (
                train.number_of_examples_per_intent[f"{intent_idx}"] == expected_count
            )

    # train exclusion will be random ...
    assert set(
        message.get(TEXT) for message in results_per_seed[0].nlu_examples
    ).intersection(message.get(TEXT) for message in results_per_seed[1].nlu_examples)
