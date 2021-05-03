from collections import ChainMap
from typing import Any, Text, Dict, List, Tuple, Optional

from rasa.shared.constants import DEFAULT_DATA_PATH
import rasa.shared.utils.common
import rasa.utils.common
import rasa.core.training
import rasa.shared.utils.io


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
        fingerprint_statuses = dict(ChainMap(*args))

        inputs_for_this_node = []

        for input, input_node in self._inputs.items():
            inputs_for_this_node.append(fingerprint_statuses[input_node])

        current_fingerprint_key = self._cache.calculate_fingerprint_key(
            self._node_name, self._config, inputs_for_this_node
        )

        fingerprint = self._cache.get_fingerprint(current_fingerprint_key)

        if fingerprint:
            fingerprint_status = FingerprintStatus(
                self._node_name,
                fingerprint,
                should_run=False,
                fingerprint_key=current_fingerprint_key,
            )

        else:
            fingerprint_status = FingerprintStatus(
                self._node_name,
                "no-fingerprint",
                should_run=True,
                fingerprint_key=current_fingerprint_key,
            )

        fingerprint_statuses[self._node_name] = fingerprint_status

        return fingerprint_statuses

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FingerprintComponent):
            return NotImplemented

        return self._node_name == other._node_name and self._config == other._config

    def __repr__(self) -> Text:
        return f"Fingerprint: {self._node_name}"


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


class TrainingCache:
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
    ) -> "Text":
        config_hash = rasa.shared.utils.io.deep_container_fingerprint(config)
        inputs_hashes = [
            rasa.shared.utils.io.deep_container_fingerprint(i) for i in inputs
        ]
        inputs_hashes = "_".join(inputs_hashes)

        return f"{node_name}_{config_hash}_{inputs_hashes}"

    def get_fingerprint(self, current_fingerprint_key: Text) -> Optional[Text]:
        return self._fingerprints.get(current_fingerprint_key)


def dask_graph_to_fingerprint_graph(
    dask_graph: Dict[Text, Tuple["RasaComponent", Text]], cache: TrainingCache
) -> Dict[Text, Tuple[FingerprintComponent, Text]]:
    fingerprint_graph = {}
    for node_name, node in dask_graph.items():
        rasa_component, *deps = node

        if not rasa_component._inputs:
            # Input nodes should always run so that we can e.g. capture changes within
            # files.
            fingerprint_graph[node_name] = (rasa_component, *deps)
        else:
            fingerprint_component = FingerprintComponent(
                config=rasa_component._config,
                node_name=rasa_component._node_name,
                inputs=rasa_component._inputs,
                cache=cache,
            )
            fingerprint_graph[node_name] = (fingerprint_component, *deps)
    return fingerprint_graph
