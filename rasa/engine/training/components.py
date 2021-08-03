from __future__ import annotations
from typing import Any, Dict, Text

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


class CachedComponent(GraphComponent):
    """Holds the cached value of a `GraphNode` from a previous training."""

    default_config = {}

    def __init__(
        self,
        cache: TrainingCache,
        output_fingerprint: Text,
        cached_node_name: Text,
        model_storage: ModelStorage,
    ):
        """Initializes a `CachedComponent`.

        Args:
            cache: The cache is used to lazily read the cached value
            output_fingerprint: Used to retrieve the cached value from the cache
            cached_node_name: Used along with the `model_storage` to load a cached
                `Resource`
            model_storage: Used to load a cached `Resource`
        """
        self._cache = cache
        self._output_fingerprint = output_fingerprint
        self._cached_node_name = cached_node_name
        self._model_storage = model_storage

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> CachedComponent:
        """Creates a new `CachedComponent` (see parent class for full docstring)."""
        return cls(
            cache=config["cache"],
            output_fingerprint=config["output_fingerprint"],
            cached_node_name=config["cached_node_name"],
            model_storage=model_storage,
        )

    def get_cached_output(self) -> Any:
        """Loads the output from the cache."""
        cached_result = self._cache.get_cached_result(self._output_fingerprint)
        if cached_result:
            return cached_result.cached_type.from_cache(
                self._cached_node_name,
                cached_result.cache_directory,
                self._model_storage,
            )
        else:
            raise ValueError("No cached output found.")
