from __future__ import annotations

import dataclasses
import uuid
from typing import Any, Dict, Optional, Text, Type

from rasa.engine.caching import Cacheable, TrainingCache
from rasa.engine.graph import ExecutionContext, GraphComponent, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training import fingerprinting


class PrecomputedValueProvider(GraphComponent):
    """Holds the precomputed values of a `GraphNode` from a previous training.

    Pre-computed values can either be
    - values loaded from cache
    - values which were provided during the fingerprint run by input nodes
    """

    def __init__(self, output: Cacheable):
        """Initializes a `PrecomputedValueProvider`.

        Args:
            output: The precomputed output to return.
        """
        self._output = output

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PrecomputedValueProvider:
        """Creates instance (see parent class for full docstring)."""
        return cls(output=config["output"])

    def get_value(self) -> Cacheable:
        """Returns the precomputed output."""
        return self._output

    @classmethod
    def replace_schema_node(cls, node: SchemaNode, output: Any) -> None:
        """Updates a `SchemaNode` to use a `PrecomputedValueProvider`.

        This is for when we want to use the precomputed output value of a node from a
        previous training in a subsequent training. We replace the class in the `uses`
        of the node to a be a `PrecomputedValueProvider` configured to return the
        precomputed value.

        Args:
            node: The node to update.
            output: precomputed cached output that the `PrecomputedValueProvider` will
            return.
        """
        node.uses = cls
        node.config = {"output": output}
        node.fn = cls.get_value.__name__
        node.constructor_name = cls.create.__name__


@dataclasses.dataclass
class FingerprintStatus:
    """Holds the output of a `FingerprintComponent` and is used to prune the graph.

    Attributes:
        output_fingerprint: A fingerprint of the node's output value.
        is_hit: `True` if node's fingerprint key exists in the cache, `False` otherwise.
    """

    output_fingerprint: Optional[Text]
    is_hit: bool

    def fingerprint(self) -> Text:
        """Returns the internal fingerprint.

        If there is no fingerprint returns a random string that will never match.
        """
        return self.output_fingerprint or uuid.uuid4().hex


class FingerprintComponent(GraphComponent):
    """Replaces non-input nodes during a fingerprint run."""

    def __init__(
        self,
        cache: TrainingCache,
        config_of_replaced_component: Dict[Text, Any],
        class_of_replaced_component: Type,
    ) -> None:
        """Initializes a `FingerprintComponent`.

        Args:
            cache: Training cache used to determine if the run is a hit or not.
            config_of_replaced_component: Needed to generate the fingerprint key.
            class_of_replaced_component: Needed to generate the fingerprint key.
        """
        self._cache = cache
        self._config_of_replaced_component = config_of_replaced_component
        self._class_of_replaced_component = class_of_replaced_component

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> FingerprintComponent:
        """Creates a `FingerprintComponent` (see parent class for full docstring)."""
        return cls(
            cache=config["cache"],
            config_of_replaced_component=config["config_of_replaced_component"],
            class_of_replaced_component=config["graph_component_class"],
        )

    def run(self, **kwargs: Any) -> FingerprintStatus:
        """Calculates the fingerprint key to determine if cached output can be used.

        If the fingerprint key matches an entry in the cache it means that there has
        been a previous node execution which matches the same component class, component
        config and input values. This means that we can potentially prune this node
        from the schema, or replace it with a cached value before the next graph run.

        Args:
            **kwargs: Inputs from all parent nodes.

        Returns:
            A `FingerprintStatus` determining if the run was a hit, and if it was a hit
            also the output fingerprint from the cache.
        """
        fingerprint_key = fingerprinting.calculate_fingerprint_key(
            graph_component_class=self._class_of_replaced_component,
            config={
                **self._class_of_replaced_component.get_default_config(),
                **self._config_of_replaced_component,
            },
            inputs=kwargs,
        )

        output_fingerprint = self._cache.get_cached_output_fingerprint(fingerprint_key)

        return FingerprintStatus(
            is_hit=output_fingerprint is not None, output_fingerprint=output_fingerprint
        )

    @classmethod
    def replace_schema_node(cls, node: SchemaNode, cache: TrainingCache) -> None:
        """Updates a `SchemaNode` to use a `FingerprintComponent`.

        This is for when we want to do a fingerprint run. During the fingerprint run we
        replace all non-input nodes with `FingerprintComponent`s so we can determine
        whether they are able to be pruned or cached before the next graph run without
        running the actual components.


        Args:
            node: The node to update.
            cache: The cache is needed to determine of there is cache hit for the
                fingerprint key.
        """
        graph_component_class = node.uses
        node.uses = cls
        # We update the node to be "eager" so that `FingerprintComponent.run` sees
        # ALL the inputs to the node. If it was not eager, we would miss any args used
        # by the constructor.
        node.eager = True
        node.constructor_name = cls.create.__name__
        node.fn = cls.run.__name__
        node.config = {
            "config_of_replaced_component": node.config,
            "cache": cache,
            "graph_component_class": graph_component_class,
        }
