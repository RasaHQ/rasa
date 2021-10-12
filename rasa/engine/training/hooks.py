import logging
from typing import Any, Dict, Text, Type

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphNodeHook
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import PrecomputedValueProvider
import rasa.shared.utils.io
from rasa.engine.training import fingerprinting

logger = logging.getLogger(__name__)


class TrainingHook(GraphNodeHook):
    """Caches fingerprints and outputs of nodes during model training."""

    def __init__(self, cache: TrainingCache, model_storage: ModelStorage):
        """Initializes a `TrainingHook`.

        Args:
            cache: Cache used to store fingerprints and outputs.
            model_storage: Used to cache `Resource`s.
        """
        self._cache = cache
        self._model_storage = model_storage

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
        fingerprint_key = fingerprinting.calculate_fingerprint_key(
            graph_component_class=graph_component_class,
            config=config,
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
        graph_component_class = self._get_graph_component_class(
            execution_context, node_name
        )
        if graph_component_class == PrecomputedValueProvider:
            return None

        output_fingerprint = rasa.shared.utils.io.deep_container_fingerprint(output)
        fingerprint_key = input_hook_data["fingerprint_key"]

        logger.debug(
            f"Caching '{output.__class__.__name__}' with fingerprint_key: "
            f"'{fingerprint_key}' and output_fingerprint '{output_fingerprint}' "
            f"calculated with data '{output}'."
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
