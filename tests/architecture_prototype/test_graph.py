import copy
import json
from typing import Any, Dict, List, Text

import rasa.architecture_prototype.model
from rasa.architecture_prototype import graph
from rasa.architecture_prototype import graph_fingerprinting
from rasa.architecture_prototype.model import Model
from rasa.architecture_prototype.graph_fingerprinting import (
    FingerprintStatus,
    TrainingCache,
)
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


def test_train_nlu():
    conftest.clean_directory()
    rasa.architecture_prototype.model._fill_defaults(nlu_train_graph_schema)
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
    def __init__(self, *args, cached_value: Any, **kwargs):
        self._cached_value = cached_value

    def get_cached_value(self, *args, **kwargs) -> Any:
        return self._cached_value


def walk_and_prune(
    graph_schema: Dict[Text, Any],
    node_name: Text,
    fingerprint_statuses: Dict[Text, FingerprintStatus],
    cache: "TrainingCache",
):
    fingerprint = fingerprint_statuses[node_name]
    should_run = fingerprint.should_run
    if not should_run:
        graph_schema[node_name]["needs"] = {}
        fingerprint_cache_key = fingerprint.fingerprint_key
        graph_schema[node_name] = {
            "uses": CachedComponent,
            "fn": "get_cached_value",
            "config": {"cached_value": cache._outputs[fingerprint_cache_key]},
            "needs": {},
        }
    else:
        for node_dependency in graph_schema[node_name]["needs"].values():
            walk_and_prune(graph_schema, node_dependency, fingerprint_statuses, cache)


def prune_graph_schema(
    graph_schema: Dict[Text, Any],
    targets: List[Text],
    fingerprint_statuses: Dict[Text, FingerprintStatus],
    cache: "TrainingCache",
) -> Dict[Text, Any]:
    graph_to_prune = copy.deepcopy(graph_schema)
    for target in targets:
        walk_and_prune(graph_to_prune, target, fingerprint_statuses, cache)

    return graph._minimal_graph_schema(graph_to_prune, targets)


def test_model_fingerprinting():
    conftest.clean_directory()
    rasa.architecture_prototype.model._fill_defaults(full_model_train_graph_schema)
    graph.visualise_as_dask_graph(full_model_train_graph_schema, "full_train_graph.png")
    core_targets = ["train_memoization_policy", "train_ted_policy", "train_rule_policy"]
    nlu_targets = [
        "train_classifier",
        "train_response_selector",
        "train_synonym_mapper",
    ]

    cache = TrainingCache()
    dask_graph = graph.convert_to_dask_graph(full_model_train_graph_schema, cache=cache)
    _ = graph.run_dask_graph(dask_graph, core_targets + nlu_targets)
    fingerprint_graph = graph_fingerprinting.dask_graph_to_fingerprint_graph(
        dask_graph, cache
    )
    fingerprint_statuses = graph.run_dask_graph(
        fingerprint_graph, core_targets + nlu_targets,
    )

    for _, fingerprint in fingerprint_statuses.items():
        if hasattr(fingerprint, "should_run"):
            assert not fingerprint.should_run

    full_model_train_graph_schema["core_train_count_featurizer1"]["config"][
        "some value"
    ] = 42
    dask_graph = graph.convert_to_dask_graph(full_model_train_graph_schema, cache=cache)
    fingerprint_graph = graph_fingerprinting.dask_graph_to_fingerprint_graph(
        dask_graph, cache
    )
    fingerprint_statuses = graph.run_dask_graph(
        fingerprint_graph, core_targets + nlu_targets
    )

    assert not fingerprint_statuses["train_classifier"].should_run
    assert fingerprint_statuses["train_ted_policy"].should_run

    pruned_graph_schema = prune_graph_schema(
        full_model_train_graph_schema,
        core_targets + nlu_targets,
        fingerprint_statuses,
        cache,
    )
    graph.visualise_as_dask_graph(pruned_graph_schema, "pruned_full_train_graph.png")

    graph.run_as_dask_graph(pruned_graph_schema, core_targets + nlu_targets)
