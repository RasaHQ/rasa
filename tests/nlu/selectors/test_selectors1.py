from pathlib import Path

import pytest
import numpy as np
from typing import List, Dict, Text, Any
from mock import Mock
from _pytest.monkeypatch import MonkeyPatch

import rasa.model
import rasa.nlu.train
from rasa.nlu.components import ComponentBuilder
from rasa.shared.nlu.training_data import util
from rasa.nlu.config import RasaNLUModelConfig
import rasa.shared.nlu.training_data.loading
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    MODEL_CONFIDENCE,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    USE_TEXT_AS_LABEL,
)
from rasa.utils import train_utils
from rasa.shared.nlu.constants import TEXT
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from tests.nlu.classifiers.test_diet_classifier import as_pipeline


def test_DIET2DIET_config_warning_transformer_with_hidden_layers():
    with pytest.warns(UserWarning) as records:
        response_selector = ResponseSelector(
            component_config={USE_TEXT_AS_LABEL: True, NUM_TRANSFORMER_LAYERS: 1}
        )

    assert len(records) > 1
    assert any(
        "We recommend to disable the hidden layers when using a transformer"
        in record.message.args[0]
        for record in records
    )
