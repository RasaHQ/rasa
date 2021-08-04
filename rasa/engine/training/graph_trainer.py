import copy
import logging
from pathlib import Path
from typing import Type

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.components import FingerprintComponent, FingerprintStatus
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

        fingerprint_results = fingerprint_graph_runner.run()

        # TODO: prune here.

        hooks = [TrainingHook(cache=self._cache, model_storage=self._model_storage)]

        graph_runner = self._graph_runner_class.create(
            graph_schema=train_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(graph_schema=train_schema),
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
                graph_schema=train_schema, model_id=model_metadata.model_id
            ),
        )
