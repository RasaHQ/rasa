import tempfile
from collections import Counter
from pathlib import Path

import pytest
import logging
from typing import List

from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from leaderboard.nlu.exp_4_stratify_infrequent_entities_only_for_exclusion import (
    Config,
    EntityExperiment,
)

logger = logging.getLogger(__file__)


def test_exclude(
    tmp_path: Path,
):

    entities = [
        (1, 2),
    ] * 10 + [(2,)] * 20

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

    experiment = EntityExperiment(
        config=Config(
            train_exclusion_fraction=0.1,
            exclusion_seed=123,
            model=None,
            data=None,
        ),
        out_dir=tmp_path,
    )

    subset = experiment.exclude_data(nlu_data)

    entities_in_subset = [
        tuple(sorted(entity["entity"] for entity in message.get(ENTITIES)))
        for message in subset.nlu_examples
    ]
    entity_counts = Counter(entities_in_subset)

    assert entity_counts[(1, 2)] == 9
    assert entity_counts[(2,)] == 18
