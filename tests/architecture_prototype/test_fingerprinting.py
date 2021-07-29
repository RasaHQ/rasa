from pathlib import Path

import rasa.architecture_prototype.model
from rasa.architecture_prototype import graph
from rasa.architecture_prototype import fingerprinting
from rasa.architecture_prototype.fingerprinting import TrainingCache
from rasa.architecture_prototype.persistence import LocalModelPersistor
from tests.architecture_prototype.conftest import project

from tests.architecture_prototype.graph_schema import train_graph_schema


def test_model_fingerprinting(tmp_path: Path):
    rasa.architecture_prototype.model.fill_defaults(train_graph_schema)
    train_graph_schema["get_project"]["config"]["project"] = project

    cache = TrainingCache()
    persistor = LocalModelPersistor(tmp_path)
    dask_graph, targets = graph.convert_to_dask_graph(
        train_graph_schema, cache=cache, model_persistor=persistor
    )
    _ = graph.run_dask_graph(dask_graph, targets)
    fingerprint_graph = fingerprinting.dask_graph_to_fingerprint_graph(dask_graph)
    fingerprint_statuses = graph.run_dask_graph(fingerprint_graph, targets,)

    for _, fingerprint in fingerprint_statuses.items():
        if hasattr(fingerprint, "should_run"):
            assert not fingerprint.should_run

    train_graph_schema["train_core_CountVectorsFeaturizer_3"]["config"][
        "some value"
    ] = 42
    dask_graph, targets = graph.convert_to_dask_graph(
        train_graph_schema, cache=cache, model_persistor=persistor
    )
    fingerprint_graph = fingerprinting.dask_graph_to_fingerprint_graph(dask_graph)
    fingerprint_statuses = graph.run_dask_graph(fingerprint_graph, targets)

    assert not fingerprint_statuses["train_DIETClassifier_5"].should_run
    assert fingerprint_statuses["TEDPolicy_1"].should_run
