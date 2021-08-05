import copy
import logging
from pathlib import Path
from typing import Any, Dict, Text, Type, Union

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import (
    CachedComponent,
    FingerprintComponent,
    FingerprintStatus,
)
from rasa.engine.training.hooks import TrainingHook
from rasa.shared.core.domain import Domain


logger = logging.getLogger(__name__)


class GraphTrainer:
    """Trains a model using a graph schema."""

    def __init__(
        self,
        model_storage: ModelStorage,
        cache: TrainingCache,
        graph_runner_class: Type[GraphRunner],
    ) -> None:
        """Initializes a `GraphTrainer`.

        Args:
            model_storage: Storage which graph components can use to persist and load.
                Also used for packaging the trained model.
            cache: Cache used to store fingerprints and outputs.
            graph_runner_class: The class to instantiate the runner from.
        """
        self._model_storage = model_storage
        self._cache = cache
        self._graph_runner_class = graph_runner_class

    def train(
        self,
        train_schema: GraphSchema,
        predict_schema: GraphSchema,
        domain_path: Path,
        output_filename: Path,
    ) -> GraphRunner:
        """Trains and packages a model then returns the prediction graph runner.

        Args:
            train_schema: The train graph schema.
            predict_schema: The predict graph schema.
            domain_path: The path to the domain file.
            output_filename: The location to save the packaged model.

        Returns:
            A graph runner loaded with the predict schema.

        """
        logger.debug("Starting training.")

        pruned_training_schema = self._fingerprint_and_prune(train_schema)

        hooks = [TrainingHook(cache=self._cache, model_storage=self._model_storage)]

        graph_runner = self._graph_runner_class.create(
            graph_schema=pruned_training_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(graph_schema=pruned_training_schema),
            hooks=hooks,
        )

        logger.debug("Train run.")

        graph_runner.run()

        domain = Domain.from_path(domain_path)
        model_metadata = self._model_storage.create_model_package(
            output_filename, train_schema, predict_schema, domain
        )

        return self._graph_runner_class.create(
            graph_schema=predict_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(
                graph_schema=predict_schema, model_id=model_metadata.model_id
            ),
        )

    def _fingerprint_and_prune(self, train_schema: GraphSchema) -> GraphSchema:
        fingerprint_schema = self._create_fingerprint_schema(train_schema)
        fingerprint_graph_runner = self._graph_runner_class.create(
            graph_schema=fingerprint_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(graph_schema=fingerprint_schema),
        )
        logger.debug("Fingerprint run.")
        fingerprint_run_outputs = fingerprint_graph_runner.run()
        pruned_training_schema = self._prune_schema(
            train_schema, fingerprint_run_outputs
        )
        return pruned_training_schema

    def _create_fingerprint_schema(self, train_schema: GraphSchema) -> GraphSchema:
        fingerprint_schema = copy.deepcopy(train_schema)
        for node_name, schema_node in fingerprint_schema.nodes.items():
            schema_node.is_target = True
            if not schema_node.is_input:
                FingerprintComponent.replace_schema_node(
                    schema_node, self._cache, node_name
                )
        return fingerprint_schema

    def _prune_schema(
        self,
        schema: GraphSchema,
        fingerprint_run_outputs: Dict[Text, Union[FingerprintStatus, Any]],
    ) -> GraphSchema:
        pruned_schema = copy.deepcopy(schema)
        target_node_names = pruned_schema.target_names()

        for target_node_name in target_node_names:
            self._walk_and_prune(
                pruned_schema, target_node_name, fingerprint_run_outputs
            )

        return pruned_schema.minimal_graph_schema()

    def _walk_and_prune(
        self,
        schema: GraphSchema,
        current_node_name: Text,
        fingerprint_run_outputs: Dict[Text, Union[FingerprintStatus, Any]],
    ) -> None:
        fingerprint_run_output = fingerprint_run_outputs[current_node_name]
        node = schema.nodes[current_node_name]
        if node.uses == CachedComponent:
            return

        if isinstance(fingerprint_run_output, FingerprintStatus):
            if fingerprint_run_output.is_hit:
                output_result = self._cache.get_cached_result(
                    output_fingerprint_key=fingerprint_run_output.output_fingerprint,
                    node_name=current_node_name,
                    model_storage=self._model_storage,
                )
                if output_result:
                    CachedComponent.replace_schema_node(node, output_result)
                    node.needs = {}
                else:
                    fingerprint_run_output.is_hit = False
        else:
            # fingerprint_run_output is just the output in this case
            # No need to run the node again.
            CachedComponent.replace_schema_node(node, fingerprint_run_output)
            node.needs = {}

        for parent_node_name in node.needs.values():
            self._walk_and_prune(schema, parent_node_name, fingerprint_run_outputs)
