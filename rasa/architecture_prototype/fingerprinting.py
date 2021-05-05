from __future__ import annotations
from collections import ChainMap
import copy
import pickle
from typing import Any, Text, Dict, List, Optional
from rasa.architecture_prototype.graph_utils import minimal_graph_schema

from rasa.architecture_prototype.interfaces import (
    GraphNodeComponent,
    GraphSchema, TrainingCacheInterface,
)
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
import rasa.shared.utils.io


class FingerprintComponent(GraphNodeComponent):
    """Represents a fingerprinted node in a graph before re-running a cached graph.

    This
    """
    def __call__(self, *args: Any) -> Dict[Text, Any]:
        fingerprint_statuses = dict(ChainMap(*args))

        inputs_for_this_node = []

        for input, input_node in self.inputs.items():
            inputs_for_this_node.append(fingerprint_statuses[input_node])

        current_fingerprint_key = self.cache.calculate_fingerprint_key(
            self.node_name, self.config, inputs_for_this_node
        )

        fingerprint = self.cache.get_fingerprint(current_fingerprint_key)

        if fingerprint:
            fingerprint_status = FingerprintStatus(
                self.node_name,
                fingerprint,
                should_run=False,
                fingerprint_key=current_fingerprint_key,
            )

        else:
            fingerprint_status = FingerprintStatus(
                self.node_name,
                "no-fingerprint",
                should_run=True,
                fingerprint_key=current_fingerprint_key,
            )

        fingerprint_statuses[self.node_name] = fingerprint_status

        return fingerprint_statuses

    @classmethod
    def from_rasa_component(cls, rasa_component) -> FingerprintComponent:
        return cls(
            config=rasa_component.config,
            node_name=rasa_component.node_name,
            inputs=rasa_component.inputs,
            cache=rasa_component.cache,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FingerprintComponent):
            return NotImplemented

        return self.node_name == other.node_name and self.config == other.config

    def __repr__(self) -> Text:
        return f"FingerprintComponent({self.node_name})"


class FingerprintStatus:
    def __init__(
        self,
        nodename: Text,
        value: str,
        should_run: Optional[bool] = None,
        fingerprint_key: Optional[Text] = None,
    ) -> None:
        self._value = value
        self._nodename = nodename
        self.should_run = should_run
        self.fingerprint_key = fingerprint_key

    def __eq__(self, other):
        if not isinstance(other, FingerprintStatus):
            return NotImplemented
        else:
            return self._value == other._value

    def fingerprint(self) -> Text:
        return self._value


class TrainingCache(TrainingCacheInterface):
    def __init__(self) -> None:
        self._fingerprints = {}
        self._outputs = {}

    def store_fingerprint(self, fingerprint_key: Text, output: Any) -> None:
        self._fingerprints[
            fingerprint_key
        ] = rasa.shared.utils.io.deep_container_fingerprint(output)

        self._outputs[fingerprint_key] = output

    def calculate_fingerprint_key(
        self, node_name: Text, config: Dict, inputs: List[Any]
    ) -> Text:
        config_hash = rasa.shared.utils.io.deep_container_fingerprint(config)
        inputs_hashes = [
            rasa.shared.utils.io.deep_container_fingerprint(i) for i in inputs
        ]
        inputs_hashes = "_".join(inputs_hashes)

        return f"{node_name}_{config_hash}_{inputs_hashes}"

    def get_fingerprint(self, current_fingerprint_key: Text) -> Optional[Text]:
        return self._fingerprints.get(current_fingerprint_key)

    def get_output(self, fingerprint_key: Text) -> Any:
        return self._outputs[fingerprint_key]

    def serialize(self) -> bytes:
        return pickle.dumps((self._fingerprints, self._outputs))

    @classmethod
    def deserialize(cls, data: bytes) -> TrainingCache:
        cache = cls()
        fingerprints, outputs = pickle.loads(data)
        cache._fingerprints = fingerprints
        cache._outputs = outputs
        return cache


def dask_graph_to_fingerprint_graph(dask_graph: DaskGraph) -> DaskGraph:
    fingerprint_graph = {}
    for node_name, node in dask_graph.items():
        rasa_component, *deps = node

        if not rasa_component.inputs:
            # Input nodes should always run so that we can e.g. capture changes within
            # files.
            fingerprint_graph[node_name] = (rasa_component, *deps)
        else:
            fingerprint_component = FingerprintComponent.from_rasa_component(
                rasa_component
            )
            fingerprint_graph[node_name] = (fingerprint_component, *deps)
    return fingerprint_graph


class CachedComponent:
    def __init__(self, *args, cached_value: Any, **kwargs):
        self._cached_value = cached_value

    def get_cached_value(self, *args, **kwargs) -> Any:
        return self._cached_value


def walk_and_prune(
    graph_schema: GraphSchema,
    node_name: Text,
    fingerprint_statuses: Dict[Text, FingerprintStatus],
    cache: TrainingCacheInterface,
):
    fingerprint = fingerprint_statuses[node_name]
    if not isinstance(fingerprint, FingerprintStatus):
        return
    should_run = fingerprint.should_run
    if not should_run:
        graph_schema[node_name]["needs"] = {}
        fingerprint_cache_key = fingerprint.fingerprint_key
        graph_schema[node_name] = {
            "uses": CachedComponent,
            "fn": "get_cached_value",
            "config": {"cached_value": cache.get_output(fingerprint_cache_key)},
            "needs": {},
        }
    else:
        for node_dependency in graph_schema[node_name]["needs"].values():
            walk_and_prune(graph_schema, node_dependency, fingerprint_statuses, cache)


def prune_graph_schema(
    graph_schema: GraphSchema,
    fingerprint_statuses: Dict[Text, FingerprintStatus],
    cache: TrainingCacheInterface,
) -> GraphSchema:
    graph_to_prune = copy.deepcopy(graph_schema)
    targets = graph_to_prune.pop("targets")
    for target in targets:
        walk_and_prune(graph_to_prune, target, fingerprint_statuses, cache)

    return minimal_graph_schema(graph_to_prune, targets)
