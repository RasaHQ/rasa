from pathlib import Path
from typing import List, Type

from rasa.engine.caching import TrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.training.hooks import TrainingHook
from rasa.shared.core.domain import Domain


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

        hooks = [TrainingHook(cache=self._cache, model_storage=self._model_storage)]

        graph_runner = self._graph_runner_class.create(
            graph_schema=train_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(graph_schema=train_schema),
            hooks=hooks,
        )

        graph_runner.run()

        domain = Domain.from_path(domain_path)
        model_metadata = self._model_storage.create_model_package(
            output_filename, train_schema, predict_schema, domain
        )

        return self._graph_runner_class.create(
            graph_schema=predict_schema,
            model_storage=self._model_storage,
            execution_context=ExecutionContext(
                graph_schema=train_schema,
                model_id=model_metadata.model_id
            ),
        )

