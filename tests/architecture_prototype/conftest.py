import shutil
from pathlib import Path
from typing import Dict, Text, Any

import pytest

import rasa.architecture_prototype.model
from rasa.architecture_prototype import graph
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
)

MODEL_DIR = "model"


def clean_directory():
    # clean up before testing persistence
    cache_dir = Path(MODEL_DIR)
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir()


@pytest.fixture
def trained_model() -> Dict:
    clean_directory()
    rasa.architecture_prototype.model._fill_defaults(full_model_train_graph_schema)
    graph.visualise_as_dask_graph(full_model_train_graph_schema, "full_train_graph.png")
    core_targets = ["train_memoization_policy", "train_ted_policy", "train_rule_policy"]
    nlu_targets = [
        "train_classifier",
        "train_response_selector",
        "train_synonym_mapper",
    ]
    trained_components = graph.run_as_dask_graph(
        full_model_train_graph_schema, core_targets + nlu_targets
    )

    return trained_components


@pytest.fixture
def prediction_graph(trained_model: Dict) -> Dict[Text, Any]:
    rasa.architecture_prototype.model._fill_defaults(predict_graph_schema)

    return predict_graph_schema
