import copy
from collections import ChainMap
import inspect
from pathlib import Path
from typing import Any, Text, Dict, List, Union, TYPE_CHECKING, Tuple, Optional

import dask

from rasa.core.channels import UserMessage
from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
import rasa.shared.utils.io

if TYPE_CHECKING:
    from rasa.core.policies.policy import PolicyPrediction


class FingerprintComponent:
    def __init__(
        self,
        config: Dict[Text, Any],
        node_name: Text,
        inputs: Dict[Text, Text],
        cache: "TrainingCache",
    ) -> None:
        self._inputs = inputs
        self._config = config
        self._node_name = node_name
        self._cache = cache

    def __call__(self, *args: Any) -> Dict[Text, Any]:
        received_inputs = dict(ChainMap(*args))

        inputs_for_this_node = []

        for input, input_node in self._inputs.items():
            inputs_for_this_node.append(received_inputs[input_node])

        fingerprint = self._cache.get_fingerprint(
            self._node_name, self._config, inputs_for_this_node
        )

        # TODO: Return result if received inputs is empty
        received_inputs[self._node_name] = fingerprint

        return received_inputs

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FingerprintComponent):
            return NotImplemented

        return self._node_name == other._node_name and self._config == other._config

    def __repr__(self) -> Text:
        return f"Fingerprint: {self._node_name}"


class Fingerprint:
    def __init__(
        self, nodename: Text, value: str, should_run: Optional[bool] = None
    ) -> None:
        self._value = value
        self._nodename = nodename
        self.should_run = should_run

    def __eq__(self, other):
        if not isinstance(other, Fingerprint):
            return NotImplemented
        else:
            return self._value == other._value

    def fingerprint(self) -> Text:
        return self._value


class TrainingCache:
    def __init__(self) -> None:
        self._fingerprints = {}

    def store_fingerprint(self, fingerprint_key: Text, output: Any) -> None:
        self._fingerprints[
            fingerprint_key
        ] = rasa.shared.utils.io.deep_container_fingerprint(output)

    def calculate_fingerprint_key(
        self, node_name: Text, config: Dict, inputs: List[Any]
    ) -> "Text":
        config_hash = rasa.shared.utils.io.deep_container_fingerprint(config)
        inputs_hashes = [
            rasa.shared.utils.io.deep_container_fingerprint(i) for i in inputs
        ]
        inputs_hashes = "_".join(inputs_hashes)

        return f"{node_name}_{config_hash}_{inputs_hashes}"

    def get_fingerprint(
        self, node_name: Text, config: Dict, inputs: List[Any]
    ) -> "Fingerprint":
        current_fingerprint_key = self.calculate_fingerprint_key(
            node_name, config, inputs
        )

        old_fingerprint = self._fingerprints.get(current_fingerprint_key)
        if old_fingerprint:
            return Fingerprint(node_name, old_fingerprint, should_run=False)

        return Fingerprint(node_name, "no-fingerprint", should_run=True)


def dask_graph_to_fingerprint_graph(
    dask_graph: Dict[Text, Tuple["RasaComponent", Text]], cache: TrainingCache
) -> Dict[Text, Tuple[FingerprintComponent, Text]]:
    fingerprint_graph = {}
    for node_name, node in dask_graph.items():
        rasa_component, *deps = node
        fingerprint_component = FingerprintComponent(
            config=rasa_component._config,
            node_name=rasa_component._node_name,
            inputs=rasa_component._inputs,
            cache=cache,
        )
        fingerprint_graph[node_name] = (fingerprint_component, *deps)
    return fingerprint_graph

