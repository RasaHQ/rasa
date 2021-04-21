import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Text

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.graph_components import load_graph_component
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
    nlu_train_graph_schema,
)
import rasa.nlu.registry
import rasa.core.registry


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

    # TODO: Fix empty message

    # TODO: should we be saving model_metadata?
    # TODO: stuff like rasa.utils.train_utils.update_deprecated_loss_type changing wont force re-train?
    # TODO: include more meta data e.g. rasa version etc.


def serialize_graph_schema(graph_schema: Dict[Text, Any]) -> Text:
    to_serialize = copy.copy(graph_schema)
    for step_name, step_config in to_serialize.items():
        component_class = step_config["uses"]
        step_config["uses"] = component_class.name
    return json.dumps(to_serialize)


def deserialize_graph_schema(serialized_graph_schema: Text) -> Dict[Text, Any]:
    schema = json.loads(serialized_graph_schema)
    for step_name, step_config in schema.items():
        component_class_name = step_config["uses"]
        try:
            step_config["uses"] = rasa.nlu.registry.get_component_class(
                component_class_name
            )
        except Exception:
            try:
                step_config["uses"] = load_graph_component(component_class_name)
            except Exception:
                try:
                    step_config["uses"] = rasa.core.registry.policy_from_module_path(
                        component_class_name
                    )
                except:
                    raise ValueError("Unknown component!")

    return schema


def test_serialize_graph_schema():
    graph.fill_defaults(full_model_train_graph_schema)
    serialized = serialize_graph_schema(full_model_train_graph_schema)
    with open("graph_schema.json", "w") as f:
        f.write(serialized)

    with open("graph_schema.json", "r") as f:
        deserialized = deserialize_graph_schema(f.read())

    core_targets = ["train_memoization_policy", "train_ted_policy", "train_rule_policy"]
    nlu_targets = [
        "train_classifier",
        "train_response_selector",
        "train_synonym_mapper",
    ]
    graph.run_as_dask_graph(deserialized, core_targets + nlu_targets)
