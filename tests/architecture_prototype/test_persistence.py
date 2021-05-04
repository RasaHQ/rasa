import rasa.architecture_prototype
from rasa.architecture_prototype import graph
from rasa.architecture_prototype.persistence import (
    serialize_graph_schema,
    deserialize_graph_schema,
)
from tests.architecture_prototype.graph_schema import full_model_train_graph_schema


def test_serialize_graph_schema():
    rasa.architecture_prototype.model._fill_defaults(full_model_train_graph_schema)
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
