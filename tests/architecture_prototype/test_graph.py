import json
import shutil
from pathlib import Path

from rasa.architecture_prototype import graph
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
    nlu_train_graph_schema,
)


def clean_directory():
    # clean up before testing persistence
    cache_dir = Path("model")
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir()


def test_train_nlu():
    clean_directory()
    graph.fill_defaults(nlu_train_graph_schema)
    serialized = json.dumps(nlu_train_graph_schema)
    deserialized = json.loads(serialized)
    graph.visualise_as_dask_graph(deserialized, "nlu_train_graph.png")
    trained_components = graph.run_as_dask_graph(
        deserialized,
        ["train_classifier", "train_response_selector", "train_synonym_mapper"],
    )
    print(trained_components)


def test_train_full_model():
    clean_directory()
    graph.fill_defaults(full_model_train_graph_schema)
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
    print(trained_components)


def test_train_load_predict():
    clean_directory()

    graph.fill_defaults(full_model_train_graph_schema)
    graph.visualise_as_dask_graph(full_model_train_graph_schema, "full_train_graph.png")
    core_targets = ["train_memoization_policy", "train_ted_policy", "train_rule_policy"]
    nlu_targets = [
        "train_classifier",
        "train_response_selector",
        "train_synonym_mapper",
    ]
    graph.run_as_dask_graph(full_model_train_graph_schema, core_targets + nlu_targets)

    graph.fill_defaults(predict_graph_schema)
    graph.visualise_as_dask_graph(predict_graph_schema, "predict_graph.png")

    predictions = graph.run_as_dask_graph(predict_graph_schema, ["select_prediction"])
    for prediction in predictions.values():
        print(prediction)

    # TODO: Metadata
    # 1. Components either return complete config or only default parameters
    # 2. Based on which we merge ourselves or merge within component
    # 3. Store full config in the graph right after building the graph

    # TODO: Fix empty message
    # TODO: Fix e2e features during prediction

    # TODO: should we be saving model_metadata?
    # TODO: stuff like rasa.utils.train_utils.update_deprecated_loss_type changing wont force re-train?




def test_serialize_graph_schema():
    assert graph.serialize_graph_schema(full_model_train_graph_schema)
