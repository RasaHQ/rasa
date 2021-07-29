from __future__ import annotations
from collections import ChainMap
import copy
import pickle
from typing import Any, Text, Dict, List, Optional
from rasa.architecture_prototype.graph_utils import minimal_graph_schema

from rasa.architecture_prototype.interfaces import (
    DaskGraph,
    GraphNodeComponent,
    GraphSchema,
    TrainingCacheInterface,
)
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
import rasa.shared.utils.io


class FingerprintComponent(GraphNodeComponent):
    """A Fingerprinted node is used to determine what parts of the graph can be cached.

    This replaces a `RasaComponent` when doing a fingerprint run of a dask graph.
    This is when we use the fingerprints stored in the `TrainingCache` to determine
    which parts of a graph need to be re-calculated on a subsequent run.
    """

    def __call__(self, *args: Any) -> Dict[Text, Any]:
        """We compare the fingerprints of the node inputs with the cache."""
        fingerprint_statuses = dict(ChainMap(*args))

        inputs_for_this_node = []

        for input, input_node in self.inputs.items():
            inputs_for_this_node.append(fingerprint_statuses[input_node])

        # Fingerprint key is a combination of node name, config and inputs.
        # This means that changing the config, or any upstream change, will result
        # in the node being re-run.
        current_fingerprint_key = self.cache.calculate_fingerprint_key(
            self.node_name, self.config, inputs_for_this_node
        )

        fingerprint = self.cache.get_fingerprint(current_fingerprint_key)

        # Fingerprint has matched. This means this component is potentially cachable
        # or prunable
        if fingerprint:
            fingerprint_status = FingerprintStatus(
                self.node_name,
                fingerprint,
                should_run=False,
                fingerprint_key=current_fingerprint_key,
            )

        # No hit means that this node will need to be re-run.
        else:
            fingerprint_status = FingerprintStatus(
                self.node_name,
                "no-fingerprint",
                should_run=True,
                fingerprint_key=current_fingerprint_key,
            )

        # Add to the fingerprint status results. The final dictionary is used to
        # prune the graph.
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
    """Stores the result of the fingerprinting run for a specific node."""

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
    """Stores the fingerprints and output values for a graph run."""

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
    """If a component can be cached it is replaced with this."""

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
    """Any component that does not have to re-run can either be:
        Pruned: if it no longer has dependencies;
        Cached: if its output is needed by other nodes.
    """
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
    """Keep, cache, or prune each node given a results of a fingerprint run."""
    graph_to_prune = copy.deepcopy(graph_schema)
    targets = graph_to_prune.pop("targets")
    for target in targets:
        walk_and_prune(graph_to_prune, target, fingerprint_statuses, cache)

    return minimal_graph_schema(graph_to_prune, targets)
