import logging
from typing import Any, Dict, Text, Type

import rasa.shared.utils.io
from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphNodeHook, GraphSchema, SchemaNode
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training import fingerprinting
from rasa.engine.training.components import PrecomputedValueProvider

logger = logging.getLogger(__name__)


class TrainingHook(GraphNodeHook):
    """Caches fingerprints and outputs of nodes during model training."""

    def __init__(
        self,
        cache: TrainingCache,
        model_storage: ModelStorage,
        pruned_schema: GraphSchema,
    ) -> None:
        """Initializes a `TrainingHook`.

        Args:
            cache: Cache used to store fingerprints and outputs.
            model_storage: Used to cache `Resource`s.
            pruned_schema: The pruned training schema.
        """
        self._cache = cache
        self._model_storage = model_storage
        self._pruned_schema = pruned_schema

    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Calculates the run fingerprint for use in `on_after_node`."""
        graph_component_class = self._get_graph_component_class(
            execution_context, node_name
        )
        graph_component_config = self._get_graph_component_config(
            execution_context, node_name
        )

        merged_config = {
            **config,
            **graph_component_config,
        }

        fingerprint_key = fingerprinting.calculate_fingerprint_key(
            graph_component_class=graph_component_class,
            config=merged_config,
            inputs=received_inputs,
        )

        return {"fingerprint_key": fingerprint_key}

    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Stores the fingerprints and caches the output of the node."""
        # We should not re-cache the output of a PrecomputedValueProvider.
        graph_component_class = self._pruned_schema.nodes[node_name].uses

        if graph_component_class == PrecomputedValueProvider:
            return None

        output_fingerprint = rasa.shared.utils.io.deep_container_fingerprint(output)
        fingerprint_key = input_hook_data["fingerprint_key"]

        logger.debug(
            f"Caching '{output.__class__.__name__}' with fingerprint_key: "
            f"'{fingerprint_key}' and output_fingerprint '{output_fingerprint}'."
        )

        self._cache.cache_output(
            fingerprint_key=fingerprint_key,
            output=output,
            output_fingerprint=output_fingerprint,
            model_storage=self._model_storage,
        )

    @staticmethod
    def _get_graph_component_class(
        execution_context: ExecutionContext, node_name: Text
    ) -> Type:
        graph_component_class = execution_context.graph_schema.nodes[node_name].uses
        return graph_component_class

    @staticmethod
    def _get_graph_component_config(
        execution_context: ExecutionContext, node_name: str
    ) -> Dict[str, Any]:
        return execution_context.graph_schema.nodes[node_name].config


class LoggingHook(GraphNodeHook):
    """Logs the training of components."""

    def __init__(self, pruned_schema: GraphSchema) -> None:
        """Creates hook.

        Args:
            pruned_schema: The pruned schema provides us with the information whether
                a component is cached or not.
        """
        self._pruned_schema = pruned_schema

    def on_before_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        received_inputs: Dict[Text, Any],
    ) -> Dict:
        """Logs the training start of a graph node."""
        node = self._pruned_schema.nodes[node_name]

        if not self._is_cached_node(node) and self._does_node_train(node):
            logger.info(f"Starting to train component '{node.uses.__name__}'.")

        return {}

    @staticmethod
    def _does_node_train(node: SchemaNode) -> bool:
        # Nodes which train are always targets so that they store their output in the
        # model storage. `is_input` filters out nodes which don't really train but e.g.
        # persist some training data.
        return node.is_target and not node.is_input

    @staticmethod
    def _is_cached_node(node: SchemaNode) -> bool:
        return node.uses == PrecomputedValueProvider

    def on_after_node(
        self,
        node_name: Text,
        execution_context: ExecutionContext,
        config: Dict[Text, Any],
        output: Any,
        input_hook_data: Dict,
    ) -> None:
        """Logs when a component finished its training."""
        node = self._pruned_schema.nodes[node_name]

        if not self._does_node_train(node):
            return

        if self._is_cached_node(node):
            actual_component = execution_context.graph_schema.nodes[node_name]
            logger.info(
                f"Restored component '{actual_component.uses.__name__}' from cache."
            )
        else:
            logger.info(f"Finished training component '{node.uses.__name__}'.")
