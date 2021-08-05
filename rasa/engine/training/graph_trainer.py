import copy
import logging
from pathlib import Path
from typing import Dict, List, Text, Type

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema, SchemaNode
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
    def __init__(
        self,
        model_storage: ModelStorage,
        cache: TrainingCache,
        graph_runner_class: Type[GraphRunner],
    ) -> None:
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
        logger.debug("Starting train.")

        fingerprint_schema = copy.deepcopy(train_schema)

        for node_name, schema_node in fingerprint_schema.nodes.items():
            schema_node.is_target = True
            if not schema_node.is_input:
                schema_node.uses = FingerprintComponent
                # TODO: We do this because otherwise FingerprintComponent does not see
                # TODO: the constructor args that come from parent nodes.
                schema_node.eager = True
                schema_node.constructor_name = "create"
                schema_node.fn = "run"
                schema_node.config.update(
                    {"cache": self._cache, "node_name": node_name,}
                )

        fingerprint_graph_runner = self._graph_runner_class.create(
            graph_schema=fingerprint_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(graph_schema=fingerprint_schema),
        )

        logger.debug("Running fingerprint run.")

        fingerprint_statuses = fingerprint_graph_runner.run()

        pruned_training_schema = self._prune_schema(train_schema, fingerprint_statuses)

        hooks = [TrainingHook(cache=self._cache, model_storage=self._model_storage)]

        graph_runner = self._graph_runner_class.create(
            graph_schema=pruned_training_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(graph_schema=pruned_training_schema),
            hooks=hooks,
        )

        logger.debug("Running main run.")

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

    def _prune_schema(
        self, schema: GraphSchema, fingerprint_statuses: Dict[Text, FingerprintStatus]
    ) -> GraphSchema:
        pruned_schema = copy.deepcopy(schema)
        target_node_names = [
            node_name
            for node_name, node in pruned_schema.nodes.items()
            if node.is_target
        ]

        for target_node_name in target_node_names:
            self._walk_and_prune(pruned_schema, target_node_name, fingerprint_statuses)

        return self._minimal_graph_schema(pruned_schema, target_node_names)

    def _walk_and_prune(
        self,
        schema: GraphSchema,
        current_node_name: Text,
        fingerprint_statuses: Dict[Text, FingerprintStatus],
    ):
        # import ipdb; ipdb.set_trace()
        fingerprint_status = fingerprint_statuses[current_node_name]
        node = schema.nodes[current_node_name]
        if node.uses == CachedComponent:
            return
        if isinstance(fingerprint_status, FingerprintStatus) and fingerprint_status.is_hit:
            output_result = self._cache.get_cached_result(
                output_fingerprint_key=fingerprint_status.output_fingerprint,
                node_name=current_node_name,
                model_storage=self._model_storage,
            )
            if output_result:
                node.needs = {}
                node.uses = CachedComponent
                node.config = {"output": output_result}
                node.fn = "get_cached_output"
                node.constructor_name = "create"
                # TODO: cached component should not cache!!!!!!!!
                # TODO: method for this?
            else:
                fingerprint_status.is_hit = False

        for parent_node_name in node.needs.values():
            self._walk_and_prune(schema, parent_node_name, fingerprint_statuses)

    # TODO: move on to graph schema
    # TODO: test it
    def _minimal_graph_schema(
        self, graph_schema: GraphSchema, targets: List[Text]
    ) -> GraphSchema:
        dependencies = self._all_dependencies_schema(graph_schema, targets)

        return GraphSchema({
            node_name: node
            for node_name, node in graph_schema.nodes.items()
            if node_name in dependencies
        })

    def _all_dependencies_schema(
        self, graph_schema: GraphSchema, targets: List[Text]
    ) -> List[Text]:
        required = []
        for target in targets:
            required.append(target)
            target_dependencies = graph_schema.nodes[target].needs.values()
            for dependency in target_dependencies:
                required += self._all_dependencies_schema(graph_schema, [dependency])

        return required
