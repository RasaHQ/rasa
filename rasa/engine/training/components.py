from __future__ import annotations
from typing import Any, Dict, Optional, Text
import dataclasses
import uuid

from rasa.engine.caching import Cacheable, TrainingCache
from rasa.engine.graph import ExecutionContext, GraphComponent, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training import fingerprinting


class CachedComponent(GraphComponent):
    """Holds the cached value of a `GraphNode` from a previous training."""

    default_config = {}

    def __init__(
        self, output: Cacheable,
    ):
        """Initializes a `CachedComponent`.

        Args:
            output: The cached output to return.
        """
        self._output = output

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CachedComponent:
        """Creates a new `CachedComponent` (see parent class for full docstring)."""
        return cls(output=config["output"],)

    def get_cached_output(self) -> Cacheable:
        """Returns the cached output."""
        return self._output

    # TODO: JUZL: should we change the node name?
    @staticmethod
    def replace_schema_node(node: SchemaNode, output):
        node.uses = CachedComponent
        node.config = {"output": output}
        node.fn = "get_cached_output"
        node.constructor_name = "create"


@dataclasses.dataclass
class FingerprintStatus:
    output_fingerprint: Optional[Text]
    is_hit: bool

    def fingerprint(self) -> Text:
        """Returns the internal fingerprint.

        If there is no fingerprint return a random string that will never match.
        """
        return self.output_fingerprint or uuid.uuid4().hex


class FingerprintComponent(GraphComponent):
    default_config = {}

    def __init__(
        self, cache: TrainingCache, component_config: Dict[Text, Any], node_name: Text
    ) -> None:
        self._cache = cache
        self._component_config = component_config
        self._node_name = node_name

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FingerprintComponent:
        cache = config.pop("cache")
        node_name = config.pop("node_name")
        return cls(cache=cache, component_config=config, node_name=node_name)

    def run(self, **kwargs: Any):
        fingerprint_key = fingerprinting.calculate_fingerprint_key(
            node_name=self._node_name,
            config=self._component_config,
            inputs=kwargs,
        )

        output_fingerprint = self._cache.get_cached_output_fingerprint(fingerprint_key)

        return FingerprintStatus(
            is_hit=output_fingerprint is not None,
            output_fingerprint=output_fingerprint
        )

    @staticmethod
    def replace_schema_node(node: SchemaNode, cache: TrainingCache, node_name: Text):
        node.uses = FingerprintComponent
        # TODO: We do this because otherwise FingerprintComponent does not see
        # TODO: the constructor args that come from parent nodes.
        node.eager = True
        node.constructor_name = "create"
        node.fn = "run"
        node.config.update(
            {"cache": cache, "node_name": node_name}
        )
