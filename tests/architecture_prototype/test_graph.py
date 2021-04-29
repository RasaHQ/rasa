import copy
import json
from typing import Any, Dict, List, Text

from rasa.architecture_prototype import graph
from rasa.architecture_prototype.graph import Fingerprint, Model, TrainingCache
from rasa.architecture_prototype.graph_components import load_graph_component
from rasa.core.channels import UserMessage
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from tests.architecture_prototype import conftest
from tests.architecture_prototype.graph_schema import (
    full_model_train_graph_schema,
    predict_graph_schema,
    nlu_train_graph_schema,
)
import rasa.nlu.registry
import rasa.core.registry


def test_train_nlu():
    conftest.clean_directory()
    graph.fill_defaults(nlu_train_graph_schema)
    serialized = json.dumps(nlu_train_graph_schema)
    deserialized = json.loads(serialized)
    graph.visualise_as_dask_graph(deserialized, "nlu_train_graph.png")
    trained_components = graph.run_as_dask_graph(
        deserialized,
        ["train_classifier", "train_response_selector", "train_synonym_mapper"],
    )
    print(trained_components)


def test_train_full_model(trained_model: Dict):
    print(trained_model)


def test_train_load_predict(prediction_graph: Dict[Text, Any]):
    graph.visualise_as_dask_graph(prediction_graph, "predict_graph.png")

    predictions = graph.run_as_dask_graph(predict_graph_schema, ["select_prediction"])
    for prediction in predictions.values():
        print(prediction)

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


def test_model_prediction_with_and_without_nlu(prediction_graph: Dict[Text, Any]):
    model = Model(prediction_graph)

    prediction_with_nlu = model.handle_message(
        DialogueStateTracker.from_events(
            "some_sender", [ActionExecuted(action_name=ACTION_LISTEN_NAME)]
        ),
        message=UserMessage(text="hello"),
    )

    prediction_without_nlu = model.predict_next_action(
        DialogueStateTracker.from_events(
            "some_sender",
            [
                ActionExecuted(action_name=ACTION_LISTEN_NAME),
                UserUttered("hi", intent={INTENT_NAME_KEY: "greet"}),
            ],
        )
    )

    assert prediction_with_nlu == prediction_without_nlu


class CachedComponent:
    def __init__(self, *args, **kwargs):
        pass

    def get_cached_value(self, *args, **kwargs):
        return "get off my lawn!"


def walk_and_prune(
    graph_schema: Dict[Text, Any],
    node_name: Text,
    fingerstatus: Dict[Text, Fingerprint],
):
    fingerprint = fingerstatus[node_name]
    should_run = fingerprint.should_run
    if not should_run:
        graph_schema[node_name]["needs"] = {}
        graph_schema[node_name] = {
            "uses": CachedComponent,
            "fn": "get_cached_value",
            "config": {"fingerprint": fingerprint},
            "needs": {},
        }
    else:
        for node_dependency in graph_schema[node_name]["needs"].values():
            walk_and_prune(graph_schema, node_dependency, fingerstatus)


def prune_graph(
    graph_schema: Dict[Text, Any],
    targets: List[Text],
    fingerstatus: Dict[Text, Fingerprint],
):
    graph_to_prune = copy.deepcopy(graph_schema)
    for target in targets:
        walk_and_prune(graph_to_prune, target, fingerstatus)

    return graph._minimal_graph_schema(graph_to_prune, targets)


def test_model_fingerprinting():
    conftest.clean_directory()
    graph.fill_defaults(full_model_train_graph_schema)
    graph.visualise_as_dask_graph(full_model_train_graph_schema, "full_train_graph.png")
    core_targets = ["train_memoization_policy", "train_ted_policy", "train_rule_policy"]
    nlu_targets = [
        "train_classifier",
        "train_response_selector",
        "train_synonym_mapper",
    ]

    cache = TrainingCache()
    _ = graph.run_as_dask_graph(
        full_model_train_graph_schema, core_targets + nlu_targets, cache=cache
    )
    finger = graph.run_as_dask_graph(
        full_model_train_graph_schema,
        core_targets + nlu_targets,
        cache=cache,
        mode="fingerprint",
    )

    for _, fingerprint in finger.items():
        assert not fingerprint.should_run

    full_model_train_graph_schema["core_train_count_featurizer1"]["config"][
        "some value"
    ] = 42

    finger = graph.run_as_dask_graph(
        full_model_train_graph_schema,
        core_targets + nlu_targets,
        cache=cache,
        mode="fingerprint",
    )

    assert not finger["train_classifier"].should_run
    assert finger["train_ted_policy"].should_run

    pruned_graph = prune_graph(
        full_model_train_graph_schema, core_targets + nlu_targets, finger
    )
    graph.visualise_as_dask_graph(pruned_graph, "pruned_full_train_graph.png")
