import tempfile
from collections import Counter
from pathlib import Path

import pytest
import logging
from typing import List

from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from leaderboard.nlu.exp_2_stratify_infrequent_entities import (
    Config,
    Experiment,
)

logger = logging.getLogger(__file__)


def test_to_train_test_split(
    tmp_path: Path,
):

    entities = [(1, 2),] * 10 + [(2,)] * 20

    messages = [
        Message(
            data={
                TEXT: f"{msg_idx}",
                INTENT: f"{msg_idx}",
                ENTITIES: [
                    {"entity": entity_idx, "value": entity_idx}
                    for entity_idx in entity_indices
                ],
            }
        )
        for msg_idx, entity_indices in enumerate(entities)
    ]
    nlu_data = TrainingData(messages)

    experiment = Experiment(
        config=Config(
            remove_data_for_entities_with_num_examples_below=0,
            train_exclusion_fraction=0.1,
            test_fraction=0.0,
            test_seed=2345,
            exclusion_seed=123,
            model=None,
            data=None,
        ),
        out_dir=tmp_path,
    )

    train, test = experiment.data_to_train_test_split(data=nlu_data)

    assert test is None

    entities_in_train = [tuple(sorted(entity['entity'] for entity in message.get(
        ENTITIES)) ) for message in train.nlu_examples]
    entity_counts = Counter(entities_in_train)

    assert entity_counts[(1,2)] == 9
    assert entity_counts[(2,)] == 18