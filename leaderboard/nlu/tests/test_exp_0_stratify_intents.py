import tempfile
from pathlib import Path

import pytest
import logging
from typing import List

from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from leaderboard.nlu.exp_0_stratify_intents import Config, IntentExperiment

logger = logging.getLogger(__file__)

# TODO: check e.g. ([4, 2], 0.1, 0., [90, 180], [10, 20]),  raises


@pytest.mark.parametrize(
    "counts, test_fraction, exclude_fraction, expected_train, expected_test, drop",
    [
        ([100, 200, 4], 0.1, 0.5, [45, 90], [10, 20], 4),
        ([100, 200, 4], 0.1, 0.0, [90, 180], [10, 20], 4),
        ([4, 2], 0.1, 0.0, [4, 2], [0, 0], 0),
    ],
)
def test_to_train_test_split(
    tmp_path: Path,
    counts: List[int],
    test_fraction: float,
    exclude_fraction: float,
    expected_train: List[int],
    expected_test: List[int],
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
                remove_data_for_intents_with_num_examples_leq=4,
                train_exclusion_fraction=exclude_fraction,
                test_fraction=test_fraction,
                test_seed=2345,  # fixed
                exclusion_seed=seed,
                model=None,
                data=None,
            ),
            out_dir=tmp_path,
        )
        train, test = experiment.to_train_test_split(nlu_data=nlu_data)
        results_per_seed.append((train, test))

        for split, expected_distribution in [
            (train, expected_train),
            (test, expected_test),
        ]:
            for intent_idx, expected_count in enumerate(expected_distribution):
                assert (
                    split.number_of_examples_per_intent[f"{intent_idx}"]
                    == expected_count
                )

    # test splits are global
    assert set(
        message.get(TEXT) for message in results_per_seed[0][1].nlu_examples
    ) == set(message.get(TEXT) for message in results_per_seed[1][1].nlu_examples)

    # train exclusion will be random ...
    assert set(
        message.get(TEXT) for message in results_per_seed[0][0].nlu_examples
    ).intersection(message.get(TEXT) for message in results_per_seed[1][0].nlu_examples)

    # ... but won't overlap with test
    for seed in [0, 1]:
        assert not set(
            message.get(TEXT) for message in results_per_seed[seed][0].nlu_examples
        ).intersection(
            message.get(TEXT) for message in results_per_seed[seed][1].nlu_examples
        )
