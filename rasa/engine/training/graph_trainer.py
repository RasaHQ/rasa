import copy
import logging
from pathlib import Path
from typing import Any, Dict, Text, Type, Union

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema, GraphModelConfiguration
from rasa.engine.constants import PLACEHOLDER_IMPORTER
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage, ModelMetadata
from rasa.engine.training.components import (
    PrecomputedValueProvider,
    FingerprintComponent,
    FingerprintStatus,
)
from rasa.engine.training.hooks import TrainingHook, LoggingHook
from rasa.shared.importers.importer import TrainingDataImporter

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

    async def train(
        self,
        model_configuration: GraphModelConfiguration,
        importer: TrainingDataImporter,
        output_filename: Path,
        force_retraining: bool = False,
        is_finetuning: bool = False,
    ) -> ModelMetadata:
        """Trains and packages a model and returns the prediction graph runner.

        Args:
            model_configuration: The model configuration (schemas, language, etc.)
            importer: The importer which provides the training data for the training.
            output_filename: The location to save the packaged model.
            force_retraining: If `True` then the cache is skipped and all components
                are retrained.
            is_finetuning: `True` if we want to finetune the model.

        Returns:
            The metadata describing the trained model.
        """
        logger.debug("Starting training.")

        # Retrieve the domain for the model metadata right at the start.
        # This avoids that something during the graph runs mutates it.
        domain = copy.deepcopy(importer.get_domain())

        if force_retraining:
            logger.debug(
                "Skip fingerprint run as a full training of the model was enforced."
            )
            pruned_training_schema = model_configuration.train_schema
        else:
            fingerprint_run_outputs = await self.fingerprint(
                model_configuration.train_schema,
                importer=importer,
                is_finetuning=is_finetuning,
            )
            pruned_training_schema = self._prune_schema(
                model_configuration.train_schema, fingerprint_run_outputs
            )

        hooks = [
            LoggingHook(pruned_schema=pruned_training_schema),
            TrainingHook(
                cache=self._cache,
                model_storage=self._model_storage,
                pruned_schema=pruned_training_schema,
            ),
        ]

        graph_runner = self._graph_runner_class.create(
            graph_schema=pruned_training_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(
                graph_schema=model_configuration.train_schema,
                is_finetuning=is_finetuning,
            ),
            hooks=hooks,
        )

        logger.debug("Running the pruned train graph with real node execution.")

        await graph_runner.run(inputs={PLACEHOLDER_IMPORTER: importer})

        return self._model_storage.create_model_package(
            output_filename, model_configuration, domain
        )

    async def fingerprint(
        self,
        train_schema: GraphSchema,
        importer: TrainingDataImporter,
        is_finetuning: bool = False,
    ) -> Dict[Text, Union[FingerprintStatus, Any]]:
        """Runs the graph using fingerprints to determine which nodes need to re-run.

        Nodes which have a matching fingerprint key in the cache can either be removed
        entirely from the graph, or replaced with a cached value if their output is
        needed by descendent nodes.

        Args:
            train_schema: The train graph schema that will be run in fingerprint mode.
            importer: The importer which provides the training data for the training.
            is_finetuning: `True` if we want to finetune the model.

        Returns:
            Mapping of node names to fingerprint results.
        """
        fingerprint_schema = self._create_fingerprint_schema(train_schema)

        fingerprint_graph_runner = self._graph_runner_class.create(
            graph_schema=fingerprint_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(
                graph_schema=train_schema, is_finetuning=is_finetuning
            ),
        )

        logger.debug("Running the train graph in fingerprint mode.")
        return await fingerprint_graph_runner.run(
            inputs={PLACEHOLDER_IMPORTER: importer}
        )

    def _create_fingerprint_schema(self, train_schema: GraphSchema) -> GraphSchema:
        fingerprint_schema = copy.deepcopy(train_schema)
        for node_name, schema_node in fingerprint_schema.nodes.items():
            # We make every node a target so that `graph_runner.run(...)` returns
            # the output for each node. We need the output of each node
            # to decide which nodes we can prune.
            schema_node.is_target = True

            # We do not replace the input nodes as we need an up-to-date fingerprint of
            # any input data to the graph. This means we can prune according to what
            # has actually changed.
            if not schema_node.is_input:
                FingerprintComponent.replace_schema_node(schema_node, self._cache)
        return fingerprint_schema

    def _prune_schema(
        self,
        schema: GraphSchema,
        fingerprint_run_outputs: Dict[Text, Union[FingerprintStatus, Any]],
    ) -> GraphSchema:
        """Uses the fingerprint statuses to prune the graph schema.

        Walks the graph starting at each target node. If a node has a cache hit we
        replace it with a `PrecomputedValueProvider` and remove its input dependencies.
        At the end, any node that is not an ancestor of a target node will be pruned
        when we call `minimal_graph_schema()`.

        Args:
            schema: The graph to prune.
            fingerprint_run_outputs: Node outputs from the fingerprint run as a mapping
                from node name to output.

        Returns:
            The pruned schema.
        """
        pruned_schema = copy.deepcopy(schema)
        target_node_names = pruned_schema.target_names

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
        """Recursively walks backwards though a graph checking the status of each node.

        If node has a fingerprint key hit then we check if there is a cached output.
        If there is a cached output we will replace the node with a
        `PrecomputedValueProvider` and remove all its dependencies (`.needs`). If
        there is not a fingerprint key hit, or there is no cached output, the node is
        left untouched and will be executed again next run unless it is no longer the
        ancestor of a target node.

        Args:
            schema: The graph we are currently walking.
            current_node_name: The current node on the walk.
            fingerprint_run_outputs: The fingerprint statuses of every node as a mapping
                from node name to status.
        """
        fingerprint_run_output = fingerprint_run_outputs[current_node_name]
        node = schema.nodes[current_node_name]

        # If we have replaced this node with a `PrecomputedValueProvider` we have
        # already visited this node. A `PrecomputedValueProvider` is updated to have
        # no parent nodes, so
        # we can end the walk here.
        if node.uses == PrecomputedValueProvider:
            return

        # If the output was a `FingerprintStatus` we must check the cache and status.
        if isinstance(fingerprint_run_output, FingerprintStatus):
            # If there is a fingerprint key hit we can potentially use a cached output.
            if fingerprint_run_output.is_hit:
                output_result = self._cache.get_cached_result(
                    output_fingerprint_key=fingerprint_run_output.output_fingerprint,
                    node_name=current_node_name,
                    model_storage=self._model_storage,
                )
                if output_result:
                    logger.debug(
                        f"Updating '{current_node_name}' to use a "
                        f"'{PrecomputedValueProvider.__name__}'."
                    )
                    PrecomputedValueProvider.replace_schema_node(node, output_result)
                    # We remove all parent dependencies as the cached output value will
                    # be used.
                    node.needs = {}
                else:
                    # If there is no cached output the node must be re-run if it ends
                    # up as an ancestor of a target node.
                    fingerprint_run_output.is_hit = False

        # Else the node was an input node and the output is the actual node's output.
        else:
            # As fingerprint_run_output is just the node's output there is no need to
            # execute the node again. We can just return it from a
            # `PrecomputedValueProvider`.
            PrecomputedValueProvider.replace_schema_node(node, fingerprint_run_output)
            node.needs = {}

        # Continue walking for every parent node.
        for parent_node_name in node.needs.values():
            self._walk_and_prune(schema, parent_node_name, fingerprint_run_outputs)
