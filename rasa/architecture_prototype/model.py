from __future__ import annotations
from typing import Text, Optional, Tuple

import dask
import dask.threaded

from rasa.architecture_prototype.graph import (
    run_as_dask_graph,
    convert_to_dask_graph,
    graph_component_for_config,
    run_dask_graph,
)
from rasa.architecture_prototype.fingerprinting import (
    dask_graph_to_fingerprint_graph,
    prune_graph_schema,
)
from rasa.architecture_prototype.graph_utils import minimal_dask_graph

from rasa.architecture_prototype.interfaces import (
    GraphSchema,
    ModelInterface,
    ModelPersistorInterface,
    TrainingCacheInterface,
)

from rasa.core.channels import UserMessage
from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


class ModelTrainer:
    """Handles the training of models using graph schema."""

    def __init__(
        self, model_persistor: ModelPersistorInterface, cache: TrainingCacheInterface,
    ) -> None:
        self._model_persistor = model_persistor
        self._cache = cache

    def train(
        self,
        project: Text,
        train_graph_schema: GraphSchema,
        predict_graph_schema: GraphSchema,
    ) -> Model:
        fill_defaults(train_graph_schema)

        # Insert project into graph
        train_graph_schema["get_project"]["config"]["project"] = project

        dask_graph, targets = convert_to_dask_graph(
            train_graph_schema, cache=self._cache, model_persistor=self._model_persistor
        )

        fingerprint_graph = dask_graph_to_fingerprint_graph(dask_graph)
        fingerprint_statuses = run_dask_graph(fingerprint_graph, targets)
        pruned_graph_schema = prune_graph_schema(
            train_graph_schema, fingerprint_statuses, self._cache,
        )

        run_as_dask_graph(
            pruned_graph_schema,
            cache=self._cache,
            model_persistor=self._model_persistor,
        )

        fill_defaults(predict_graph_schema)

        return Model(
            predict_graph_schema,
            train_graph_schema,
            self._cache,
            persistor=self._model_persistor,
        )


class Model(ModelInterface):
    """A model uses a dask graph to predict actions based on user messages."""

    def __init__(
        self,
        predict_graph_schema: GraphSchema,
        train_graph_schema: GraphSchema,
        cache: TrainingCacheInterface,
        persistor: ModelPersistorInterface,
    ) -> None:
        self._persistor = persistor
        self._predict_graph_schema = predict_graph_schema
        self._dask_graph, self._predict_graph_targets = convert_to_dask_graph(
            self._predict_graph_schema, model_persistor=persistor
        )
        self._train_graph_schema = train_graph_schema
        self._cache = cache

    def handle_message(
        self, tracker: DialogueStateTracker, message: Optional[UserMessage]
    ) -> Tuple[PolicyPrediction, Optional[UserUttered]]:
        graph = self._dask_graph.copy()

        # Insert user message into graph
        graph["load_user_message"] = graph_component_for_config(
            "load_user_message",
            self._predict_graph_schema["load_user_message"],
            {"message": message},
        )

        # Insert dialogue history into graph
        graph["load_history"] = graph_component_for_config(
            "load_history",
            self._predict_graph_schema["load_history"],
            {"tracker": tracker},
        )

        results = run_dask_graph(graph, ["select_prediction", "add_parsed_nlu_message"])

        user_event = None
        if message:
            updated_tracker: DialogueStateTracker = results["add_parsed_nlu_message"]
            user_event = updated_tracker.get_last_event_for(event_type=UserUttered)

        return results["select_prediction"], user_event

    def predict_next_action(
        self, tracker: DialogueStateTracker,
    ) -> Tuple[PolicyPrediction, UserUttered]:
        return self.handle_message(tracker, message=None)

    def get_domain(self) -> Domain:
        domain_graph = minimal_dask_graph(self._dask_graph, targets=["load_domain"])
        return dask.threaded.get(domain_graph, "load_domain")["load_domain"]

    def persist(self, target: Text) -> None:
        self._persistor.create_model_package(target, self)

    @classmethod
    def load(cls, target: Text, persistor: ModelPersistorInterface) -> Model:
        return persistor.load_model_package(target)

    def train(self, project: Text) -> Model:
        trainer = ModelTrainer(self._persistor, self._cache)
        return trainer.train(
            project, self._train_graph_schema, self._predict_graph_schema
        )


def fill_defaults(graph_schema: GraphSchema):
    """Fills a graph schema with default config values for each component."""

    for step_name, step_config in graph_schema.items():
        if step_name == "targets":
            continue
        component_class = step_config["uses"]

        if hasattr(component_class, "defaults"):
            step_config["config"] = {
                **component_class.defaults,
                **step_config["config"],
            }
