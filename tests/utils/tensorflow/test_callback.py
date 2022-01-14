from typing import Dict, Text
from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory

from rasa.core import training

from rasa.core.policies.ted_policy import TEDPolicy
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.shared.core.domain import Domain
from rasa.utils.tensorflow.callback import RasaModelCheckpoint
from rasa.utils.tensorflow.constants import EPOCHS


@pytest.mark.parametrize(
    "previous_best, current_values, improved",
    [
        (
            {"val_i_acc": 0.5, "val_f1": 0.5},
            {"val_i_acc": 0.65, "val_f1": 0.7},
            True,
        ),  # both improved
        (
            {"val_i_acc": 0.54, "val_f1": 0.5},
            {"val_i_acc": 0.54, "val_f1": 0.7},
            True,
        ),  # one equal, one improved
        (
            {"val_i_acc": 0.8, "val_f1": 0.55},
            {"val_i_acc": 0.8, "val_f1": 0.55},
            False,
        ),  # both equal
        (
            {"val_i_acc": 0.64, "val_f1": 0.5},
            {"val_i_acc": 0.41, "val_f1": 0.7},
            False,
        ),  # one improved, one worse
        (
            {"val_i_acc": 0.71, "val_f1": 0.35},
            {"val_i_acc": 0.52, "val_f1": 0.35},
            False,
        ),  # one worse, one equal
    ],
)
def test_does_model_improve(
    previous_best: Dict[Text, float],
    current_values: Dict[Text, float],
    improved: bool,
    tmpdir: Path,
):
    checkpoint = RasaModelCheckpoint(tmpdir)
    checkpoint.best_metrics_so_far = previous_best
    # true iff all values are equal or better and at least one is better
    assert checkpoint._does_model_improve(current_values) == improved


@pytest.fixture(scope="module")
def trained_ted(
    tmp_path_factory: TempPathFactory, moodbot_domain_path: Path
) -> TEDPolicy:
    training_files = "data/test_moodbot/data/stories.yml"
    domain = Domain.load(moodbot_domain_path)
    trackers = training.load_data(str(training_files), domain)
    policy = TEDPolicy.create(
        {**TEDPolicy.get_default_config(), EPOCHS: 1},
        LocalModelStorage.create(tmp_path_factory.mktemp("storage")),
        Resource("ted"),
        ExecutionContext(GraphSchema({})),
    )
    policy.train(trackers, domain)

    return policy


@pytest.mark.parametrize(
    "previous_best, current_values, improved",
    [
        ({"val_i_acc": 0.5, "val_f1": 0.5}, {"val_i_acc": 0.5, "val_f1": 0.7}, True),
        ({"val_i_acc": 0.5, "val_f1": 0.5}, {"val_i_acc": 0.4, "val_f1": 0.5}, False),
    ],
)
def test_on_epoch_end_saves_checkpoints_file(
    previous_best: Dict[Text, float],
    current_values: Dict[Text, float],
    improved: bool,
    tmp_path: Path,
    trained_ted: TEDPolicy,
):
    model_name = "checkpoint"
    best_model_file = tmp_path / model_name
    assert not best_model_file.exists()
    checkpoint = RasaModelCheckpoint(tmp_path)
    checkpoint.best_metrics_so_far = previous_best
    checkpoint.model = trained_ted.model
    checkpoint.on_epoch_end(1, current_values)
    if improved:
        assert best_model_file.exists()
    else:
        assert not best_model_file.exists()
