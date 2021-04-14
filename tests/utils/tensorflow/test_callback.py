from typing import Dict, Text, List
from pathlib import Path

import pytest

from rasa.core.agent import Agent
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.utils.tensorflow.callback import RasaModelCheckpoint

from tests.core.conftest import form_bot_agent

BOTH_IMPROVED = [{"val_i_acc": 0.5, "val_f1": 0.5}, {"val_i_acc": 0.7, "val_f1": 0.7}]
ONE_IMPROVED_ONE_EQUAL = [
    {"val_i_acc": 0.5, "val_f1": 0.5},
    {"val_i_acc": 0.5, "val_f1": 0.7},
]
BOTH_EQUAL = [{"val_i_acc": 0.7, "val_f1": 0.7}, {"val_i_acc": 0.7, "val_f1": 0.7}]
ONE_IMPROVED_ONE_WORSE = [
    {"val_i_acc": 0.6, "val_f1": 0.5},
    {"val_i_acc": 0.4, "val_f1": 0.7},
]
ONE_WORSE_ONE_EQUAL = [
    {"val_i_acc": 0.7, "val_f1": 0.5},
    {"val_i_acc": 0.5, "val_f1": 0.5},
]


@pytest.mark.parametrize(
    "logs, improved",
    [
        (BOTH_IMPROVED, True),
        (ONE_IMPROVED_ONE_EQUAL, True),
        (BOTH_EQUAL, False),
        (ONE_IMPROVED_ONE_WORSE, False),
        (ONE_WORSE_ONE_EQUAL, False),
    ],
)
def test_does_model_improve(logs: List[Dict[Text, float]], improved, tmpdir: Path):
    checkpoint = RasaModelCheckpoint(tmpdir)
    checkpoint.best_metrics_so_far = logs[0]
    # true iff all values are equal or better and at least one is better
    assert checkpoint._does_model_improve(logs[1]) == improved


@pytest.fixture(scope="session")
def form_bot_ted(form_bot_agent: Agent) -> TEDPolicy:
    for policy in form_bot_agent.policy_ensemble.policies:
        if isinstance(policy, TEDPolicy):
            return policy


@pytest.mark.parametrize(
    "logs, improved", [(BOTH_IMPROVED, True), (ONE_WORSE_ONE_EQUAL, False),]
)
def test_on_epoch_end_saves_checkpoints_file(
    logs: List[Dict[Text, float]], improved: bool, form_bot_ted: TEDPolicy, tmpdir: Path
):
    model_name = "checkpoint"
    best_model_file = Path(str(tmpdir), model_name)
    assert not best_model_file.exists()
    checkpoint = RasaModelCheckpoint(tmpdir)
    checkpoint.best_metrics_so_far = logs[0]
    checkpoint.model = form_bot_ted.model
    checkpoint.on_epoch_end(1, logs[1])
    if improved:
        assert best_model_file.exists()
    else:
        assert not best_model_file.exists()
