import rasa.architecture_prototype.model
from rasa.architecture_prototype import graph
from rasa.architecture_prototype import fingerprinting
from rasa.architecture_prototype.fingerprinting import TrainingCache

from tests.architecture_prototype.graph_schema import train_graph_schema


def test_model_fingerprinting():
    rasa.architecture_prototype.model.fill_defaults(train_graph_schema)

    cache = TrainingCache()
    dask_graph, targets = graph.convert_to_dask_graph(train_graph_schema, cache=cache)
    _ = graph.run_dask_graph(dask_graph, targets)
    fingerprint_graph = fingerprinting.dask_graph_to_fingerprint_graph(
        dask_graph, cache
    )
    fingerprint_statuses = graph.run_dask_graph(fingerprint_graph, targets,)

    for _, fingerprint in fingerprint_statuses.items():
        if hasattr(fingerprint, "should_run"):
            assert not fingerprint.should_run

    train_graph_schema["core_train_count_featurizer1"]["config"]["some value"] = 42
    dask_graph, targets = graph.convert_to_dask_graph(train_graph_schema, cache=cache)
    fingerprint_graph = fingerprinting.dask_graph_to_fingerprint_graph(
        dask_graph, cache
    )
    fingerprint_statuses = graph.run_dask_graph(fingerprint_graph, targets)

    assert not fingerprint_statuses["train_classifier"].should_run
    assert fingerprint_statuses["train_ted_policy"].should_run
