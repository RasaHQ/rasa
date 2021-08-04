from __future__ import annotations
from typing import Any, Dict, Text

from rasa.engine.caching import Cacheable
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


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
